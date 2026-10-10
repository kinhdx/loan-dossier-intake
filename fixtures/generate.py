"""Generates the synthetic evaluation dossiers in fixtures/dossiers/.

Each dossier has an application form and a payslip as text PDFs, and a citizen ID card as an image-only
PDF (rendered, noised and JPEG-compressed so that it has no text layer). Every person, employer and
number is invented. expected.json records the values as written on each document, the defect planted in
the dossier and the decision a correct pipeline should reach.

The expected findings are declared here, from the defect that was planted. They are not computed by the
pipeline's checks, so the evaluation can catch a check that is wrong.

    uv run python fixtures/generate.py
"""
from __future__ import annotations

import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pymupdf

OUT = Path(__file__).parent / "dossiers"
AS_OF = date(2026, 10, 1)
SIGNED = AS_OF - timedelta(days=2)
SEED = 20261001
RATE = 0.18
FONT, FONT_BOLD = "notos", "notosbo"

FAMILY = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương"]
MIDDLE_M = ["Văn", "Minh", "Quang", "Đức", "Hữu", "Thanh"]
MIDDLE_F = ["Thị", "Thu", "Ngọc", "Thanh", "Thùy", "Mai"]
GIVEN_M = ["Nam", "Tuấn", "Dũng", "Phong", "Khoa", "Hải", "Long", "Quân", "Thắng"]
GIVEN_F = ["Hương", "Lan", "Hà", "Trang", "Linh", "Mai", "Yến", "Thảo", "Ngân"]
EMPLOYERS = ["Công ty TNHH Phần mềm Sao Mai", "Công ty CP Thực phẩm An Phú", "Công ty TNHH Vận tải Hưng Thịnh",
             "Công ty CP Dược phẩm Bình An", "Công ty TNHH Điện tử Minh Quang", "Công ty CP Xây dựng Đông Á"]
PROVINCES = ["001", "079", "031", "048", "036", "024"]
CITIES = {"001": "Hà Nội", "079": "TP. Hồ Chí Minh", "031": "Hải Phòng", "048": "Đà Nẵng", "036": "Nam Định",
          "024": "Bắc Giang"}

# Stress set: low-resolution blurred ID scans, scanned forms with a corrected amount, and payslips that put the
# year-to-date column before the monthly one.
DEFECTS_HARD = ["clean"] * 5 + ["NAME_MISMATCH", "ID_EXPIRED", "INCOME_ABOVE_PAYSLIP", "DTI_ABOVE_LIMIT", "ID_MISMATCH"]
DEFECTS = (["clean"] * 11 + ["NAME_MISMATCH"] * 2 + ["ID_MISMATCH"] * 2 + ["DOB_MISMATCH"] + ["ID_EXPIRED"] * 2
           + ["ID_NUMBER_INCONSISTENT"] * 2 + ["INCOME_ABOVE_PAYSLIP"] * 2 + ["EMPLOYER_MISMATCH"]
           + ["DTI_ABOVE_LIMIT"] * 2 + ["AGE_OUT_OF_RANGE", "AMOUNT_OUT_OF_RANGE", "INCOME_BELOW_MINIMUM",
                                       "MISSING_DOCUMENT", "PAYSLIP_NAME_MISMATCH"])


def parse_amount_written(text: str) -> int:
    """Inverse of the formats used on the form: '150.000.000 đồng' or '150 triệu đồng'."""
    number = text.split()[0]
    return int(number) * 1_000_000 if "triệu" in text else int(number.replace(".", ""))


def money(n: int) -> str:
    return f"{n:,}".replace(",", ".") + " đồng"


def dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def installment(amount: int, term: int) -> float:
    r = RATE / 12
    return amount * r / (1 - (1 + r) ** -term)


def id_number(rng: random.Random, province: str, dob: date, female: bool) -> str:
    century_digit = 2 * ((dob.year - 1900) // 100) + (1 if female else 0)
    return f"{province}{century_digit}{dob.year % 100:02d}{rng.randrange(10**6):06d}"


def card_expiry(dob: date) -> date:
    """A citizen ID card is valid until the holder's next 25th, 40th or 60th birthday."""
    return next(dob.replace(year=dob.year + age) for age in (25, 40, 60) if dob.replace(year=dob.year + age) > AS_OF)


def person(rng: random.Random, age_range=(25, 45), dob: date | None = None) -> dict:
    female = rng.random() < 0.5
    name = " ".join([rng.choice(FAMILY), rng.choice(MIDDLE_F if female else MIDDLE_M),
                     rng.choice(GIVEN_F if female else GIVEN_M)])
    age = rng.randint(*age_range)
    dob = dob or date(AS_OF.year - age - 1, rng.randint(1, 12), rng.randint(1, 28))
    province = rng.choice(PROVINCES)
    return {"name": name, "female": female, "dob": dob, "province": province,
            "id": id_number(rng, province, dob, female),
            "expiry": card_expiry(dob),
            "phone": "09" + "".join(str(rng.randrange(10)) for _ in range(8))}


def salary(rng: random.Random, net_target: int) -> dict:
    gross = round(net_target / 0.86, -5)
    insurance = round(gross * 0.105, -3)
    tax = round(max(0, (gross - insurance - 11_000_000) * 0.05), -3)
    allowance = round(gross * 0.15, -5)
    return {"gross": int(gross), "basic": int(gross - allowance), "allowance": int(allowance),
            "insurance": int(insurance), "tax": int(tax), "net": int(gross - insurance - tax)}


def loan(rng: random.Random, net: int, debt: int, age: int) -> tuple[int, int, int]:
    """Amount, term and existing debt inside every product limit, with DTI at most 45%.

    If no loan fits next to the existing debt (low income, high debt), the debt is dropped and the draw repeated.
    """
    terms = [t for t in (12, 24, 36, 48, 60) if age + t / 12 <= 59]
    for attempt in range(400):
        if attempt == 200:
            debt = 0
        term = rng.choice(terms)
        amount = int(round(rng.uniform(20_000_000, min(500_000_000, 12 * net)), -6))
        if (debt + installment(amount, term)) / net <= 0.45:
            return amount, term, debt
    raise ValueError(f"no loan fits net income {net}")


def build_truth(rng: random.Random, defect: str, hard: bool = False) -> dict:
    dob = None
    if defect == "ID_EXPIRED":  # holder turned 25 or 40 a few weeks or months ago and has not renewed the card
        birthday = AS_OF - timedelta(days=rng.randint(10, 200))
        dob = birthday.replace(year=birthday.year - rng.choice([25, 40]), day=min(birthday.day, 28))
    p = person(rng, (56, 58) if defect == "AGE_OUT_OF_RANGE" else (25, 45), dob)
    net_target = rng.randrange(10_000_000, 40_000_000, 500_000)
    if defect == "INCOME_BELOW_MINIMUM":
        net_target = 7_400_000
    if defect == "AMOUNT_OUT_OF_RANGE":
        net_target = 62_000_000
    pay = salary(rng, net_target)
    debt = rng.choice([0, 0, 1_500_000, 2_000_000, 3_000_000])
    age = AS_OF.year - p["dob"].year - 1
    amount, term, debt = loan(rng, pay["net"], debt, age)
    employer = rng.choice(EMPLOYERS)

    form = {"full_name": p["name"], "date_of_birth": dmy(p["dob"]), "id_number": p["id"], "phone": p["phone"],
            "employer": employer, "declared_monthly_income": money(pay["net"]), "existing_monthly_debt": money(debt),
            "requested_amount": money(amount), "term_months": f"{term} tháng"}
    card = {"full_name": p["name"].upper(), "date_of_birth": dmy(p["dob"]), "id_number": p["id"],
            "sex": "Nữ" if p["female"] else "Nam", "expiry_date": dmy(p["expiry"])}
    slip = {"employee_name": p["name"], "employer": employer.upper(), "pay_period": f"{AS_OF.month - 1:02d}/{AS_OF.year}",
            "gross_income": money(pay["gross"]), "net_income": money(pay["net"])}

    if defect == "NAME_MISMATCH":
        parts = p["name"].split()
        pool = [m for m in (MIDDLE_F if p["female"] else MIDDLE_M) if m != parts[1]]
        form["full_name"] = " ".join([parts[0], rng.choice(pool), parts[2]])
    elif defect == "PAYSLIP_NAME_MISMATCH":
        slip["employee_name"] = person(rng)["name"]
    elif defect == "ID_MISMATCH":
        digits = list(p["id"])
        digits[8], digits[10] = str((int(digits[8]) + 3) % 10), str((int(digits[10]) + 7) % 10)
        form["id_number"] = "".join(digits)
    elif defect == "DOB_MISMATCH":
        shifted = p["dob"].day + 3 if p["dob"].day <= 25 else p["dob"].day - 3  # same month and year
        form["date_of_birth"] = dmy(p["dob"].replace(day=shifted))
    elif defect == "ID_EXPIRED":
        expired = max(p["dob"].replace(year=p["dob"].year + age) for age in (25, 40)
                      if p["dob"].replace(year=p["dob"].year + age) < AS_OF)
        card["expiry_date"] = dmy(expired)
    elif defect == "ID_NUMBER_INCONSISTENT":
        wrong = id_number(rng, p["province"], p["dob"], not p["female"])
        card["id_number"] = form["id_number"] = wrong
    elif defect == "INCOME_ABOVE_PAYSLIP":
        form["declared_monthly_income"] = money(int(round(pay["net"] * 1.3, -5)))
    elif defect == "EMPLOYER_MISMATCH":
        form["employer"] = rng.choice([e for e in EMPLOYERS if e != employer])
    elif defect == "DTI_ABOVE_LIMIT":
        term = 12
        amount = int(round(pay["net"] * rng.uniform(8, 11), -6))
        form["requested_amount"], form["term_months"] = money(amount), f"{term} tháng"
    elif defect == "AGE_OUT_OF_RANGE":
        form["term_months"] = "60 tháng"
        amount = int(round(min(amount, pay["net"] * 10), -6))
        form["requested_amount"] = money(amount)
    elif defect == "AMOUNT_OUT_OF_RANGE":
        form["requested_amount"] = money(600_000_000)
        form["term_months"] = "60 tháng"

    style = {"form": rng.choice(["lines", "table"]), "payslip": rng.choice(["lines", "table"]),
             "form_scanned": rng.random() < 0.2}
    if rng.random() < 0.5:  # amounts in millions, as many applicants write them
        for key in ("requested_amount", "declared_monthly_income"):
            value = int(form[key].replace(".", "").split()[0])
            form[key] = f"{round(value / 1_000_000)} triệu đồng"
    if rng.random() < 0.4:
        months = int(form["term_months"].split()[0])
        if months % 12 == 0:
            form["term_months"] = f"{months // 12} năm"
    if rng.random() < 0.4:
        d, m, y = (int(x) for x in form["date_of_birth"].split("/"))
        form["date_of_birth"] = f"{d}/{m}/{y}"

    if hard:
        written = parse_amount_written(form["requested_amount"])
        form["requested_amount"] = money(written)
        style = {"form": "lines", "payslip": "table_ytd_first", "form_scanned": True,
                 "correction": money(written + rng.choice([50, 80, 100]) * 1_000_000)}

    expected_codes = [] if defect in ("clean",) else [defect]
    decision = ("incomplete" if defect == "MISSING_DOCUMENT" else
                "ready_for_underwriting" if defect == "clean" else "needs_review")
    docs = {"don-de-nghi-vay.pdf": ("application_form", form), "cccd.pdf": ("national_id", card),
            "phieu-luong.pdf": ("payslip", slip)}
    if defect == "MISSING_DOCUMENT":
        del docs["phieu-luong.pdf"]
    return {"as_of": AS_OF.isoformat(), "defect": defect, "expected_findings": expected_codes,
            "expected_decision": decision, "salary": pay, "province": CITIES[p["province"]], "style": style,
            "documents": {name: {"doc_type": t, "fields": f} for name, (t, f) in docs.items()}}


def _page(size=(595, 842)):
    doc = pymupdf.open()
    return doc, doc.new_page(width=size[0], height=size[1])


def _finish(doc) -> bytes:
    doc.subset_fonts()
    return doc.tobytes(garbage=3, deflate=True)


def text_page(lines: list[tuple[str, float, bool]]) -> bytes:
    doc, page = _page()
    y = 60
    font = pymupdf.Font(FONT)
    for row in lines:
        text, fontsize, bold = row[:3]
        if text:
            page.insert_text((56, y), text, fontname=FONT_BOLD if bold else FONT, fontsize=fontsize)
        if len(row) == 5:  # (label, size, bold, struck value, corrected value)
            x = 56 + font.text_length(text, fontsize)
            struck_width = font.text_length(row[3], fontsize)
            page.insert_text((x, y), row[3], fontname=FONT, fontsize=fontsize)
            page.draw_line((x - 1, y - fontsize * 0.3), (x + struck_width + 1, y - fontsize * 0.3), width=1.1)
            page.insert_text((x + struck_width + 10, y), row[4], fontname=FONT, fontsize=fontsize)
        y += fontsize * 1.9
    return _finish(doc)


def table_page(title: list[str], rows: list[tuple[str, ...]], widths: list[float], footer: list[str]) -> bytes:
    """Labels and values in separate columns with ruled rows, as on many bank and payroll forms."""
    doc, page = _page()
    y = 60
    for i, line in enumerate(title):
        page.insert_text((56, y), line, fontname=FONT_BOLD, fontsize=14 if i == 0 else 11)
        y += 24
    y += 6
    for row in rows:
        page.draw_line((56, y - 13), (540, y - 13), color=(0.6, 0.6, 0.6), width=0.5)
        x = 60
        for j, cell in enumerate(row):
            page.insert_text((x, y), cell, fontname=FONT_BOLD if j == 0 and cell.isupper() else FONT, fontsize=10)
            x += widths[j]
        y += 22
    page.draw_line((56, y - 13), (540, y - 13), color=(0.6, 0.6, 0.6), width=0.5)
    y += 14
    for line in footer:
        page.insert_text((56, y), line, fontname=FONT, fontsize=10)
        y += 18
    return _finish(doc)


PURPOSES = ["Sửa chữa nhà ở", "Mua sắm đồ dùng gia đình", "Chi phí học tập", "Chi phí y tế"]
FORM_LABELS = [("Họ và tên", "full_name"), ("Ngày sinh", "date_of_birth"), ("Số CCCD", "id_number"),
               ("Điện thoại", "phone"), ("Đơn vị công tác", "employer"),
               ("Thu nhập ròng hằng tháng", "declared_monthly_income"),
               ("Nghĩa vụ trả nợ hiện tại mỗi tháng", "existing_monthly_debt"),
               ("Số tiền đề nghị vay", "requested_amount"), ("Thời hạn vay", "term_months")]


def application_pdf(rng: random.Random, f: dict, style: str, correction: str = "") -> bytes:
    purpose = rng.choice(PURPOSES)
    consent = ["Tôi cam kết thông tin kê khai là đúng sự thật và đồng ý cho Ngân hàng xử lý dữ liệu cá nhân",
               "của tôi cho mục đích thẩm định khoản vay.",
               f"Ngày {SIGNED:%d} tháng {SIGNED:%m} năm {SIGNED:%Y}", "Người đề nghị (ký, ghi rõ họ tên)"]
    if style == "table":
        rows = [(label, f[key]) for label, key in FORM_LABELS] + [("Mục đích vay", purpose)]
        return table_page(["ĐƠN ĐỀ NGHỊ VAY VỐN TIÊU DÙNG", "Kính gửi: Ngân hàng (mẫu)"], rows, [210, 270], consent)
    rows = [("ĐƠN ĐỀ NGHỊ VAY VỐN TIÊU DÙNG", 15, True), ("Kính gửi: Ngân hàng (mẫu)", 11, False), ("", 6, False),
            ("I. THÔNG TIN KHÁCH HÀNG", 12, True)]
    rows += [(f"{label}: {f[key]}", 11, False) for label, key in FORM_LABELS[:7]]
    rows += [("", 6, False), ("II. ĐỀ NGHỊ VAY", 12, True)]
    for label, key in FORM_LABELS[7:]:
        if key == "requested_amount" and correction:
            rows.append((f"{label}: ", 11, False, correction, f[key]))
        else:
            rows.append((f"{label}: {f[key]}", 11, False))
    rows += [(f"Mục đích vay: {purpose}", 11, False), ("", 6, False), ("III. CAM KẾT", 12, True)]
    rows += [(consent[0], 10, False), (consent[1], 10, False), ("", 6, False), (consent[2], 11, False),
             (consent[3], 11, False)]
    return text_page(rows)


def payslip_pdf(f: dict, pay: dict, style: str) -> bytes:
    month = int(f["pay_period"].split("/")[0])
    if style == "table_ytd_first":
        rows = [("KHOẢN MỤC", f"LŨY KẾ NĂM {AS_OF.year}", f"THÁNG {f['pay_period']}"),
                ("Lương cơ bản", money(pay["basic"] * month), money(pay["basic"])),
                ("Phụ cấp", money(pay["allowance"] * month), money(pay["allowance"])),
                ("Tổng thu nhập", money(pay["gross"] * month), f["gross_income"]),
                ("Bảo hiểm bắt buộc (10,5%)", money(pay["insurance"] * month), money(pay["insurance"])),
                ("Thuế thu nhập cá nhân", money(pay["tax"] * month), money(pay["tax"])),
                ("Thực lĩnh", money(pay["net"] * month), f["net_income"])]
        return table_page([f["employer"], "BẢNG THANH TOÁN LƯƠNG CÁ NHÂN",
                           f"Nhân viên: {f['employee_name']}    Kỳ lương: {f['pay_period']}"],
                          rows, [170, 160, 150], ["Phòng Nhân sự - Kế toán"])
    if style == "table":
        # Year-to-date column: plausible but wrong values for an extractor that reads the wrong column.
        rows = [("KHOẢN MỤC", f"THÁNG {f['pay_period']}", f"LŨY KẾ NĂM {AS_OF.year}"),
                ("Lương cơ bản", money(pay["basic"]), money(pay["basic"] * month)),
                ("Phụ cấp", money(pay["allowance"]), money(pay["allowance"] * month)),
                ("Tổng thu nhập", f["gross_income"], money(pay["gross"] * month)),
                ("Bảo hiểm bắt buộc (10,5%)", money(pay["insurance"]), money(pay["insurance"] * month)),
                ("Thuế thu nhập cá nhân", money(pay["tax"]), money(pay["tax"] * month)),
                ("Thực lĩnh", f["net_income"], money(pay["net"] * month))]
        return table_page([f["employer"], "BẢNG THANH TOÁN LƯƠNG CÁ NHÂN",
                           f"Nhân viên: {f['employee_name']}    Kỳ lương: {f['pay_period']}"],
                          rows, [170, 150, 160], ["Phòng Nhân sự - Kế toán"])
    rows = [(f["employer"], 12, True), (f"PHIẾU LƯƠNG THÁNG {f['pay_period']}", 15, True), ("", 6, False),
            (f"Họ và tên nhân viên: {f['employee_name']}", 11, False), ("Phòng ban: Vận hành", 11, False),
            ("", 4, False), (f"Lương cơ bản: {money(pay['basic'])}", 11, False),
            (f"Phụ cấp: {money(pay['allowance'])}", 11, False),
            (f"Tổng thu nhập: {f['gross_income']}", 11, True),
            (f"Bảo hiểm bắt buộc (10,5%): {money(pay['insurance'])}", 11, False),
            (f"Thuế thu nhập cá nhân: {money(pay['tax'])}", 11, False),
            (f"Thực lĩnh: {f['net_income']}", 12, True), ("", 6, False),
            ("Phiếu lương được phát hành bởi phòng Nhân sự - Kế toán.", 9, False)]
    return text_page(rows)


def id_card_face(f: dict, province: str) -> bytes:
    doc, page = _page((242.6, 153.0))  # 85.6 x 54 mm
    page.draw_rect(page.rect, color=(0.55, 0.62, 0.75), fill=(0.93, 0.95, 0.98), width=1.2)
    page.draw_rect(pymupdf.Rect(10, 42, 62, 112), color=(0.5, 0.5, 0.5), fill=(0.82, 0.84, 0.86))
    lines = [(70, 14, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", 5.6, True), (88, 21, "Độc lập - Tự do - Hạnh phúc", 5, False),
             (82, 33, "CĂN CƯỚC CÔNG DÂN", 8, True), (70, 46, f"Số / No.: {f['id_number']}", 7, True),
             (70, 58, "Họ và tên / Full name:", 5, False), (70, 67, f["full_name"], 7, True),
             (70, 79, f"Ngày sinh / Date of birth: {f['date_of_birth']}", 5.5, False),
             (70, 89, f"Giới tính / Sex: {f['sex']}    Quốc tịch / Nationality: Việt Nam", 5.5, False),
             (70, 99, f"Quê quán / Place of origin: {province}", 5.5, False),
             (70, 109, f"Nơi thường trú / Place of residence: {province}", 5.5, False),
             (10, 135, f"Có giá trị đến / Date of expiry: {f['expiry_date']}", 5.5, False)]
    for x, y, text, size, bold in lines:
        page.insert_text((x, y), text, fontname=FONT_BOLD if bold else FONT, fontsize=size)
    return _finish(doc)


def scan(pdf: bytes, rng: random.Random, dpi: int, blur: bool = False, quality=(35, 70)) -> bytes:
    """Image-only copy of page 1: slightly rotated, dimmed, noisy and JPEG-compressed, with no text layer."""
    src = pymupdf.open(stream=pdf, filetype="pdf")
    w, h = src[0].rect.width, src[0].rect.height
    canvas = pymupdf.open()
    page = canvas.new_page(width=w * 1.08, height=h * 1.08)
    page.draw_rect(page.rect, color=None, fill=(0.97, 0.96, 0.93))
    page.show_pdf_page(pymupdf.Rect(w * 0.04, h * 0.04, w * 1.04, h * 1.04), src, 0, rotate=rng.uniform(-3.5, 3.5))
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(8, 18), img.shape)
    if blur:  # 3x3 box blur, as from a phone camera slightly out of focus
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        img = sum(padded[dy:dy + img.shape[0], dx:dx + img.shape[1]] for dy in range(3) for dx in range(3)) // 9
    img = np.clip(img * rng.uniform(0.8, 1.0) + noise + rng.randint(-10, 10), 0, 255).astype(np.uint8)
    noisy = pymupdf.Pixmap(pymupdf.csRGB, pix.width, pix.height, img.tobytes(), False)
    jpg = noisy.tobytes("jpg", jpg_quality=rng.randint(*quality))
    out = pymupdf.open()
    sheet = out.new_page(width=page.rect.width, height=page.rect.height)
    sheet.insert_image(sheet.rect, stream=jpg)
    return out.tobytes(deflate=True)


def write_set(prefix: str, defects: list[str], seed: int, hard: bool) -> None:
    rng = random.Random(seed)
    defects = defects[:]
    rng.shuffle(defects)
    for i, defect in enumerate(defects, start=1):
        truth = build_truth(rng, defect, hard)
        folder = OUT / f"{prefix}{i:03d}"
        folder.mkdir(parents=True)
        docs, style = truth["documents"], truth["style"]
        form = application_pdf(rng, docs["don-de-nghi-vay.pdf"]["fields"], style["form"], style.get("correction", ""))
        if style["form_scanned"]:
            form = scan(form, rng, dpi=110)
        (folder / "don-de-nghi-vay.pdf").write_bytes(form)
        card = id_card_face(docs["cccd.pdf"]["fields"], truth["province"])
        if hard:
            card_pdf = scan(card, rng, dpi=100, blur=True, quality=(22, 32))
        else:
            card_pdf = scan(card, rng, dpi=rng.choice([180, 220, 260]))
        (folder / "cccd.pdf").write_bytes(card_pdf)
        if "phieu-luong.pdf" in docs:
            slip = payslip_pdf(docs["phieu-luong.pdf"]["fields"], truth["salary"], style["payslip"])
            (folder / "phieu-luong.pdf").write_bytes(slip)
        (folder / "expected.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    write_set("D", DEFECTS, SEED, hard=False)
    write_set("H", DEFECTS_HARD, SEED + 1, hard=True)
    print(f"wrote {len(DEFECTS)} standard and {len(DEFECTS_HARD)} stress dossiers to {OUT}")


if __name__ == "__main__":
    main()
