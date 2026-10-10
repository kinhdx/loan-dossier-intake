# 0001: Rules decide the route, the model only reads

## Context

A dossier either goes straight to an underwriter or to a reviewer first. The model can read a document well,
but its own statement of confidence is not calibrated: a misread digit on a blurred ID card comes back with
the same confidence as a correct one. A routing decision in a bank also has to be explained to an auditor
after the fact.

## Decision

The model classifies documents and copies fields as written. Routing is decided by the rules in
`checks.py`: completeness, field formats, grounding of each value in the document text, agreement between
documents, ID card rules and product limits. Every rule that fails produces a named finding, and the dossier
decision is the most severe finding.

Scanned documents have no text layer to ground values in. For them the rules rely on redundancy that
already exists in a loan dossier: the same name, ID number and date of birth appear on the form and the card,
and the CCCD number itself encodes the holder's century, sex and birth year.

## Consequences

- A misread usually produces a contradiction, so it lands in review instead of passing silently. The
  evaluation counts the cases where it does not (`ready_with_wrong_data`).
- More dossiers go to review than strictly necessary when scans are poor. That cost is visible in the
  false review rate and is the number to bring down, by better capture or a stronger model, not by relaxing
  rules.
- Every rule is unit-tested against the synthetic dossiers with perfect extraction, so a rule change that
  breaks an expected outcome fails CI.
