"""Convert the sample spreadsheet (Sample_JSON.xlsx) into the JSON document the API takes.

usage: python scripts/xlsx_to_json.py Sample_JSON.xlsx out.json
"""
import json
import sys

import openpyxl

KEEP = ["id", "question", "answer", "comments", "confidence"]


def convert(src, dst):
    ws = openpyxl.load_workbook(src, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(rows)]

    records = []
    for r in rows:
        rec = {h: v for h, v in zip(header, r) if h in KEEP and v not in (None, "")}
        if rec.get("question"):  # sheet has ~1000 rows, most are blank
            records.append(rec)

    with open(dst, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} records to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
