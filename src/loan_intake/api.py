"""HTTP API for intake, the review queue and policy questions.

    POST /dossiers                 upload the files of one dossier; returns the decision and findings
    GET  /dossiers/{id}            full result, including extracted fields and reviewer actions
    GET  /review-queue             dossiers waiting for a reviewer (incomplete ones go back to the customer)
    POST /dossiers/{id}/review     accept or return a dossier, with the reviewer's note; an incomplete
                                   dossier cannot be accepted (409)
    POST /ask                      answer a question from the policy documents

ID numbers are masked in every response except GET /dossiers/{id}, which is the case-handling view.
"""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from loan_intake.cli import DB_PATH, _index
from loan_intake.documents import UnreadableDocument, load
from loan_intake.llm import AnthropicClient, ModelClient
from loan_intake.normalize import mask_id
from loan_intake.pipeline import Settings, process
from loan_intake.rag.answer import ask
from loan_intake.schemas import Decision
from loan_intake.store import Store

app = FastAPI(title="Loan dossier intake")


@lru_cache
def get_store() -> Store:
    return Store(DB_PATH)


@lru_cache
def get_client() -> ModelClient:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    return AnthropicClient()


@lru_cache
def get_index():
    return _index()


class ReviewIn(BaseModel):
    reviewer: str
    action: Literal["accept", "return"]
    note: str = ""


class AskIn(BaseModel):
    question: str


@app.post("/dossiers")
def create_dossier(files: list[UploadFile], store: Store = Depends(get_store),
                   client: ModelClient = Depends(get_client)) -> dict:
    # Sync handler: FastAPI runs it in a worker thread, so the model calls do not block the event loop.
    try:
        sources = [load(f.filename or "upload", f.file.read()) for f in files]
    except UnreadableDocument as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    dossier_id = uuid.uuid4().hex[:12]
    result = process(dossier_id, sources, Settings(client))
    store.save(result)
    return {
        "id": dossier_id, "decision": result.decision.value,
        "findings": [{"code": f.code, "severity": f.severity.value, "document": f.document, "field": f.field,
                      "message": f.message} for f in result.findings],
        "documents": [{"name": d.name, "type": d.doc_type.value, "scanned": d.scanned,
                       "id_number": mask_id(d.get("id_number")) if d.get("id_number") else None}
                      for d in result.documents],
        "metrics": result.metrics,
    }


@app.get("/dossiers/{dossier_id}")
def get_dossier(dossier_id: str, store: Store = Depends(get_store)) -> dict:
    record = store.get(dossier_id)
    if record is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    return record


@app.get("/review-queue")
def review_queue(store: Store = Depends(get_store)) -> list[dict]:
    return store.queue()


@app.post("/dossiers/{dossier_id}/review")
def review(dossier_id: str, body: ReviewIn, store: Store = Depends(get_store)) -> dict:
    record = store.get(dossier_id)
    if record is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    if body.action == "accept" and record["decision"] == Decision.INCOMPLETE:
        raise HTTPException(status_code=409, detail="dossier is missing a required document and cannot be accepted")
    store.review(dossier_id, body.reviewer, body.action, body.note)
    return {"id": dossier_id, "status": "ready" if body.action == "accept" else "returned"}


@app.post("/ask")
def ask_policy(body: AskIn, client: ModelClient = Depends(get_client)) -> dict:
    store, embedder = get_index()
    answer = ask(body.question, store, embedder, client, Settings(client).extract_model)
    return {"answerable": answer.answerable, "answer": answer.text, "citations": answer.citations,
            "retrieved": answer.retrieved}
