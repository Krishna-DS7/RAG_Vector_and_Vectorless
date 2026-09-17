"""
shared/sample.py

A small built-in document, used when no file is given, so the tool runs with
no dependencies and nothing to download.

It is a fake 14-page policy handbook, and every ugly thing in it is something
real PDF extraction produces:

  1. a running header on every page
  2. a footer with a changing page number
  3. ligature glyphs        "office" stored as o + one "ffi" glyph + ce
  4. non-breaking spaces    "60,000\u00a0units"
  5. curly quotes, em-dashes
  6. hard line wraps every ~45 characters
  7. a word split by hyphen        "calen-\\ndar"
  8. a REAL hyphen also split      "one-\\ntime"     ambiguous, no rule solves it
  9. a sentence running across a page break        pages 9 -> 10
 10. a contents page whose lines look exactly like section headings

Point the tool at your own PDF to see how it copes with real damage:

    python3 parse.py yourfile.pdf
"""

SOURCE_NAME = "sample_policy_handbook.pdf"

PAGES = [
    (1, """ACME ROBOTICS

EMPLOYEE POLICY HANDBOOK

Version 4.1 \u2014 Effective 1 April 2024

Confidential \u2014 Page 1 of 14"""),

    (2, """Acme Robotics \u2014 Employee Policy Handbook

CONTENTS

A.1  Leave ................................. 3
A.1.1  Annual Leave ........................ 3
A.1.2  Sick Leave .......................... 5
A.2  Pay and Expenses ...................... 6
A.2.1  Annual Bonus ........................ 6
A.2.2  Expense Reimbursement ............... 8
A.3  Remote Work ........................... 11
A.3.1  Eligibility ......................... 11
A.3.2  Equipment Allowance ................. 12

Confidential \u2014 Page 2 of 14"""),

    (3, """Acme Robotics \u2014 Employee Policy Handbook

A.1  LEAVE

A.1.1  Annual Leave

Full-time employees receive 24 days of
annual leave each calen-
dar year, accrued monthly. Leave requests
must be submitted through the HR portal at
least 10 working days in advance.

Confidential \u2014 Page 3 of 14"""),

    (4, """Acme Robotics \u2014 Employee Policy Handbook

Unused leave up to 6 days may be carried
into the next year. Public holidays are
additional to annual leave. Leave taken
during a notice period must be approved in
writing by HR.

Confidential \u2014 Page 4 of 14"""),

    (5, """Acme Robotics \u2014 Employee Policy Handbook

A.1.2  Sick Leave

Employees may take up to 12 days of paid
sick leave per year. For any absence longer
than 3 consecutive days a medical certi\ufb01cate
from a registered doctor is required.

Confidential \u2014 Page 5 of 14"""),

    (6, """Acme Robotics \u2014 Employee Policy Handbook

A.2  PAY AND EXPENSES

A.2.1  Annual Bonus

The annual bonus pool is funded at 10 percent
of salary cost and is paid each June.

Confidential \u2014 Page 6 of 14"""),

    (7, """Acme Robotics \u2014 Employee Policy Handbook

Individual bonus depends on company
performance and your rating. Employees who
join after 1 October are not eligible in that
bonus year.

Confidential \u2014 Page 7 of 14"""),

    (8, """Acme Robotics \u2014 Employee Policy Handbook

A.2.2  Expense Reimbursement

To claim a work expense, submit the receipt
through the expense portal within 30 days of
the spend. Claims over 5,000\u00a0units require
manager approval before the cost is incurred.

Confidential \u2014 Page 8 of 14"""),

    # ---- ends MID-SENTENCE; continues on page 10 ----
    (9, """Acme Robotics \u2014 Employee Policy Handbook

Reimbursement is paid with the next salary
run. Travel, client meals and professional
subscriptions may be claimed as expenses.
Home of\ufb01ce equipment is NOT claimed through this

Confidential \u2014 Page 9 of 14"""),

    (10, """Acme Robotics \u2014 Employee Policy Handbook

process; see the equipment allowance in
section A.3.2. Claims submitted without a
receipt are rejected.

Confidential \u2014 Page 10 of 14"""),

    (11, """Acme Robotics \u2014 Employee Policy Handbook

A.3  REMOTE WORK

A.3.1  Eligibility

Employees may work remotely up to 3 days per
week after completing probation. Your home
working location must be declared to HR.

Confidential \u2014 Page 11 of 14"""),

    # ---- "one-\ntime" must survive as "one-time" ----
    (12, """Acme Robotics \u2014 Employee Policy Handbook

A.3.2  Equipment Allowance

Each remote employee receives a one-
time home of\ufb01ce equipment allowance of
60,000\u00a0units.

Confidential \u2014 Page 12 of 14"""),

    (13, """Acme Robotics \u2014 Employee Policy Handbook

The allowance covers a desk, an ergonomic
chair, a monitor, a keyboard and a headset.
It refreshes every 4 years.

Confidential \u2014 Page 13 of 14"""),

    (14, """Acme Robotics \u2014 Employee Policy Handbook

Order equipment through the internal
catalogue; do not pay for it your-
self. Within the allowance a desk may cost up
to 25,000\u00a0units. Equipment remains company
property and is returned when you leave.

Confidential \u2014 Page 14 of 14"""),
]
