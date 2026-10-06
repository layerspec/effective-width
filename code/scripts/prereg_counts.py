"""Count the verdict column of paper/table_prereg.tex (Table S17), the one source every quoted total must match.

Usage: python scripts/prereg_counts.py [--table ../paper/table_prereg.tex]
"""
import argparse, collections, re

CANON = [  # (regex on the verdict cell, canonical label)
    (r"^holds in reverse|^reversed", "reversed"),
    (r"^holds by letter", "holds by letter only"),
    (r"^holds", "holds"),
    (r"^first clause holds", "partly"),
    (r"^partly|^\(a\)", "partly"),
    (r"^fails", "fails"),
    (r"^pending", "pending"),
    (r"^cancelled", "cancelled"),
    (r"^not judged|^no threshold|^descriptive|^reported", "reported only"),
    (r"^not found", "not found"),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", default="../paper/table_prereg.tex"); a = ap.parse_args()
    counts = collections.Counter(); rows = 0
    for line in open(a.table, encoding="utf-8"):
        if not re.match(r"^(P|H|S|A|B|C|§)", line) or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split(" & ")]
        if len(cells) < 5:
            continue
        rows += 1; v = cells[-2]
        label = next((lab for rx, lab in CANON if re.search(rx, v)), "other: " + v[:30])
        counts[label] += 1
    print(f"Table S17: {rows} registered statements")
    for k, n in counts.most_common():
        print(f"  {n:4d}  {k}")

if __name__ == "__main__":
    main()
