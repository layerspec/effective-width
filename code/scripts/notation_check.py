"""Notation consistency scan of paper/main.tex (submission bar B13).

Two checks, both mechanical:
  1. one spelling per symbol: every symbol below has a canonical form and a list of
     rival spellings; any rival that occurs is reported with its line numbers;
  2. defined before used: the first non-comment line on which a symbol occurs must not
     precede the notation table (\\label{tab:notation}) unless the symbol is on the
     allow list of things the introduction may use in words (k^*, tau, r_max, C, PR).
Exit status 1 if either check finds something, so `make notation` can gate a build.

    python scripts/notation_check.py --tex ../paper/main.tex
"""
from __future__ import annotations

import argparse
import re
import sys

# canonical regex -> rival regexes
SYMBOLS = {
    r"r_\{\\max\}": [r"r_\{\\rm max\}", r"r_\{\\mathrm\{max\}\}", r"r_\{\\text\{max\}\}", r"r_\\max\b", r"\br_max\b"],
    r"k\^\*": [r"k\^\\star", r"k\^\{\*\}", r"k\^\{\\star\}", r"k_\*"],
    r"C_\{\\rm out\}": [r"C_\{\\text\{out\}\}", r"C_\{\\mathrm\{out\}\}", r"C_\{out\}", r"C_\\text\{out\}"],
    r"C_\{\\rm in\}": [r"C_\{\\text\{in\}\}", r"C_\{\\mathrm\{in\}\}", r"C_\{in\}"],
    r"C_\{\\rm class\}": [r"C_\{\\text\{class\}\}", r"C_\{\\mathrm\{class\}\}"],
    r"\\mathrm\{PR\}": [r"\\text\{PR\}", r"\\textrm\{PR\}", r"\\operatorname\{PR\}"],
    r"\\mathrm\{erank\}": [r"\\text\{erank\}", r"\\operatorname\{erank\}"],
    r"\\Sigma_\{\\rm patch\}": [r"\\Sigma_\{\\text\{patch\}\}", r"\\Sigma_\{\\mathrm\{patch\}\}", r"\\Sigma_\{patch\}"],
    r"\\Sigma_\{\\rm out\}": [r"\\Sigma_\{\\text\{out\}\}", r"\\Sigma_\{\\mathrm\{out\}\}", r"\\Sigma_\{out\}"],
    r"\\Sigma_\{\\rm act\}": [r"\\Sigma_\{\\text\{act\}\}", r"\\Sigma_\{\\mathrm\{act\}\}"],
    r"\\rho_\{\\rm depth\}": [r"\\rho_\{\\text\{depth\}\}", r"\\rho_\{\\mathrm\{depth\}\}", r"\\rho_\{depth\}"],
    r"\\rho_\{\\rm align\}": [r"\\rho_\{\\text\{align\}\}", r"\\rho_\{\\mathrm\{align\}\}", r"\\rho_\{align\}", r"\\rho_\\text\{align\}"],
    r"W_\{\\circ\}": [r"W_\\circ\b", r"W_o\b", r"W_\{o\}", r"W_\{\\rm o\}"],
    r"k_h k_w": [r"k_h\\,k_w", r"k_hk_w", r"k_h \\cdot k_w"],   # k_h \times k_w is the kernel size, a different thing
    r"\\operatorname\{tr\}": [r"\\mathrm\{tr\}", r"\\text\{tr\}", r"\\mathrm\{Tr\}"],
    r"\\operatorname\{rank\}": [r"\\mathrm\{rank\}", r"\\text\{rank\}"],
    r"\|\\cdot\|_\{\\rm op\}": [r"\|_\{\\text\{op\}\}", r"\|_\{\\mathrm\{op\}\}", r"\|_2\b"],
    r"depthwise": [r"depth-wise", r"depth wise"],
    r"checkpoint": [r"check-point", r"check point"],
    r"top-\$k\$": [r"top-k\b"],   # "the top $k$ eigenvalues" is prose, allowed
    r"batch-norm": [r"batchnorm", r"batch norm\b"],
}
ALLOW_BEFORE_TABLE = {r"k\^\*", r"r_\{\\max\}", r"\\mathrm\{PR\}", r"C_\{\\rm out\}", r"C_\{\\rm in\}", r"C_\{\\rm class\}",
                      r"\\Sigma_\{\\rm patch\}", r"k_h k_w", r"depthwise", r"checkpoint", r"batch-norm", r"top-\$k\$"}
STYLE_RIVALS = [r"_\{\\text\{[A-Za-z]+\}\}", r"_\{\\mathrm\{[A-Za-z]+\}\}"]   # the paper uses _{\rm word}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tex", default="../paper/main.tex")
    a = p.parse_args(argv)
    lines = open(a.tex).read().split("\n")
    body = [(i, l) for i, l in enumerate(lines, 1) if not l.lstrip().startswith("%")]
    table_line = next((i for i, l in body if "\\label{tab:notation}" in l), None)
    problems = 0

    def occurrences(rx):
        return [i for i, l in body if re.search(rx, l)]

    print(f"{a.tex}: {len(lines)} lines; notation table at line {table_line}")
    print("\n1. rival spellings")
    for canon, rivals in SYMBOLS.items():
        n = len(occurrences(canon))
        for r in rivals:
            occ = occurrences(r)
            if occ:
                problems += 1
                print(f"   {canon!s:28s} ({n} uses)  rival {r!s:30s} x{len(occ)} at lines {occ[:8]}")
    for rx in STYLE_RIVALS:
        hits = {}
        for i, l in body:
            for m in re.findall(rx, l):
                hits.setdefault(m, []).append(i)
        for m, occ in sorted(hits.items()):
            problems += 1
            print(f"   style: {m} x{len(occ)} at lines {occ[:8]}  (paper style is _{{\\rm word}})")
    print("   none" if problems == 0 else "")

    print("\n2. used before the notation table")
    before = 0
    if table_line:
        for canon in SYMBOLS:
            occ = occurrences(canon)
            if occ and occ[0] < table_line and canon not in ALLOW_BEFORE_TABLE:
                before += 1
                print(f"   {canon!s:28s} first at line {occ[0]} (< {table_line})")
    print("   none" if before == 0 else "")
    problems += before
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
