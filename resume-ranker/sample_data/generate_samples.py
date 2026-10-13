#!/usr/bin/env python3
"""
sample_data/generate_samples.py — builds 5 realistic sample resume PDFs
(with reportlab) + 2 sample job descriptions, for 2 different roles.

Role A (accountant, retail shop):
  1. priya_sharma_accountant.pdf  — strong fit (should rank #1 for JD A)
  2. rahul_verma_fresher.pdf      — partial fit (B.Com fresher, no Tally/GST)
  3. amit_kumar_electrician.pdf   — mismatch (should rank last for JD A)
Role B (delivery executive):
  4. suresh_yadav_driver.pdf      — strong fit (should rank #1 for JD B)
  5. neha_gupta_assistant.pdf     — weak fit (no licence, office background)

Names/phones are fictional. Run:  python sample_data/generate_samples.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

STYLES = getSampleStyleSheet()
TITLE = STYLES["Title"]
H2 = STYLES["Heading2"]
BODY = STYLES["BodyText"]
BODY.fontSize = 10
BODY.leading = 14


def make_pdf(filename, name, sections):
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    story = [Paragraph(name, TITLE), Spacer(1, 6)]
    for heading, lines in sections:
        story.append(Paragraph(heading, H2))
        for line in lines:
            story.append(Paragraph(line, BODY))
        story.append(Spacer(1, 4))
    doc.build(story)
    print("wrote", filename)


JD_ACCOUNTANT = """Accountant — Sharma General Store, Noida

We are a retail general store hiring a full-time Accountant.

Requirements:
- B.Com degree, minimum 2 years of accounting experience in retail
- Strong hands-on knowledge of Tally Prime for daily billing and bookkeeping
- GST filing: monthly GSTR-1 and GSTR-3B returns, invoicing with GST
- MS Excel for stock records and sales reports (pivot tables a plus)
- Bank reconciliation and petty cash handling

Salary: Rs 22,000-28,000/month. Immediate joiners preferred.
"""

JD_DRIVER = """Delivery Executive — FreshKart Groceries, Noida

We are hiring delivery executives for grocery delivery across Noida sectors.

Requirements:
- Valid two-wheeler license (LMV license a plus)
- Minimum 1 year of food delivery or courier experience
- Own smartphone; comfortable with delivery apps and route planning
- Good knowledge of Noida city routes
- Basic vehicle maintenance

Salary: Rs 18,000/month + delivery incentives. Two-wheeler provided on request.
"""

RESUMES = [
    ("priya_sharma_accountant.pdf", "Priya Sharma", [
        ("Contact", ["+91 98765 11111 · priya.sharma.example@gmail.com · Noida, UP"]),
        ("Objective", ["B.Com graduate with 3 years of retail accounting experience, "
                       "seeking an Accountant role. Strong in Tally Prime, GST filing and Excel."]),
        ("Experience", [
            "<b>Accountant — Gupta Departmental Store, Noida</b> (2022–2025)",
            "Daily billing and bookkeeping in Tally Prime; monthly GST filing "
            "(GSTR-1, GSTR-3B); invoicing; bank reconciliation; petty cash handling.",
            "Maintained stock records and monthly sales reports in MS Excel using "
            "pivot tables; assisted with TDS computation and audit preparation.",
        ]),
        ("Education", ["B.Com, Delhi University (2021)"]),
        ("Skills", ["Tally Prime, GST filing, GSTR-1, GSTR-3B, invoicing, billing, "
                    "bookkeeping, bank reconciliation, petty cash, MS Excel, pivot table, TDS"]),
    ]),
    ("rahul_verma_fresher.pdf", "Rahul Verma", [
        ("Contact", ["+91 98765 22222 · rahul.verma.example@gmail.com · Greater Noida, UP"]),
        ("Objective", ["B.Com fresher looking for an entry-level accounts or data entry role. "
                       "Fast learner, good with computers."]),
        ("Experience", [
            "<b>Intern — City Computer Centre, Greater Noida</b> (6 months, 2024)",
            "Data entry of sales records; typing at 40 wpm; prepared quotations in MS Word; "
            "email handling and front desk support.",
        ]),
        ("Education", ["B.Com, CCS University (2024)"]),
        ("Skills", ["MS Excel, data entry, typing, MS Word, email handling, front desk"]),
    ]),
    ("amit_kumar_electrician.pdf", "Amit Kumar", [
        ("Contact", ["+91 98765 33333 · amit.kumar.example@gmail.com · Noida, UP"]),
        ("Objective", ["ITI-certified electrician with 8 years of experience in residential "
                       "and commercial electrical work."]),
        ("Experience", [
            "<b>Electrician — Bright Home Services, Noida</b> (2017–2025)",
            "Electrical wiring for new constructions; fault diagnosis and repair; "
            "CCTV installation; generator maintenance; AC repair assistance.",
        ]),
        ("Education", ["ITI Electrician trade, Govt. ITI Noida (2017)"]),
        ("Skills", ["electrician, electrical wiring, CCTV installation, "
                    "generator maintenance, AC repair, customer service"]),
    ]),
    ("suresh_yadav_driver.pdf", "Suresh Yadav", [
        ("Contact", ["+91 98765 44444 · suresh.yadav.example@gmail.com · Noida, UP"]),
        ("Objective", ["Delivery professional with 5 years of food delivery and courier "
                       "experience across Noida and Delhi."]),
        ("Experience", [
            "<b>Delivery Partner — food delivery & courier apps, Noida</b> (2020–2025)",
            "Last mile delivery of 60+ orders/day; route planning across Noida sectors; "
            "cash handling on COD orders; basic vehicle maintenance of two-wheeler.",
        ]),
        ("Licences", ["Valid two-wheeler license; LMV license (2023). Own smartphone."]),
        ("Skills", ["two-wheeler license, LMV license, last mile delivery, food delivery, "
                    "courier, route planning, vehicle maintenance, cash handling, customer service"]),
    ]),
    ("neha_gupta_assistant.pdf", "Neha Gupta", [
        ("Contact", ["+91 98765 55555 · neha.gupta.example@gmail.com · Noida, UP"]),
        ("Objective", ["Office assistant with 4 years of experience in computer operation "
                       "and front-office work, looking for delivery or field roles."]),
        ("Experience", [
            "<b>Office Assistant — Agarwal Traders, Noida</b> (2021–2025)",
            "Computer operator: data entry in MS Excel and Google Sheets; maintained "
            "purchase orders; front desk and customer support over phone.",
        ]),
        ("Education", ["B.A., IGNOU (2020)"]),
        ("Skills", ["computer operator, data entry, MS Excel, Google Sheets, purchase orders, "
                    "front desk, customer support, Hindi, English"]),
    ]),
]


def main():
    for filename, name, sections in RESUMES:
        make_pdf(filename, name, sections)
    (OUT / "jd_accountant.txt").write_text(JD_ACCOUNTANT, encoding="utf-8")
    (OUT / "jd_driver.txt").write_text(JD_DRIVER, encoding="utf-8")
    print("wrote jd_accountant.txt, jd_driver.txt")


if __name__ == "__main__":
    main()
