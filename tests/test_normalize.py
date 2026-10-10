from datetime import date

from loan_intake.normalize import fold, mask_id, parse_amount, parse_date, parse_int, parse_term_months
from loan_intake.policy import monthly_installment


def test_parse_amount_vietnamese_formats():
    assert parse_amount("15.000.000 đồng") == 15_000_000
    assert parse_amount("15,000,000 VND") == 15_000_000
    assert parse_amount("8.500.000đ") == 8_500_000
    assert parse_amount("15 triệu đồng") == 15_000_000
    assert parse_amount("15,5 triệu") == 15_500_000
    assert parse_amount("1.5 triệu") == 1_500_000
    assert parse_amount("1.500 triệu") == 1_500_000_000
    assert parse_amount("không rõ") is None
    assert parse_amount("") is None


def test_parse_date_day_first():
    assert parse_date("05/03/1990") == date(1990, 3, 5)
    assert parse_date("Ngày sinh: 5-3-1990") == date(1990, 3, 5)
    assert parse_date("31/02/1990") is None


def test_parse_int_term():
    assert parse_int("36 tháng") == 36
    assert parse_int("") is None
    assert parse_term_months("3 năm") == 36
    assert parse_term_months("24 tháng") == 24
    assert parse_term_months("5 năm.") == 60
    assert parse_term_months("36") == 36
    assert parse_term_months("3 năm 6 tháng") is None
    assert parse_term_months("ba mươi sáu tháng") is None


def test_fold_removes_diacritics_and_case():
    assert fold("  NGUYỄN  Thị Hương ") == "nguyen thi huong"
    assert fold("Đặng Đức") == "dang duc"


def test_mask_id_keeps_three_digits_each_end():
    assert mask_id("001190012345") == "001******345"


def test_installment_matches_annuity_formula():
    # 120,000,000 over 12 months at 18%/year: about 11,001,000 per month
    assert round(monthly_installment(120_000_000, 12, 0.18), -3) == 11_002_000
