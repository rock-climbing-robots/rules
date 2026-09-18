#!/usr/bin/env python3
"""Inject genuine ArUco bit patterns (IDs 1-5) into the figures.

Each marker is a 6x6 cell grid (4x4 data bits + 1-cell black border) from the
DICT_4X4_50 dictionary.  For every ID, the list of *white* cells is written as
a TikZ macro \\arucoA ... \\arucoE (IDs 1 ... 5), cells given as col/row with
row 0 at the BOTTOM (TikZ y up).  Patterns are checked by rendering each marker
and detecting it back with OpenCV.
"""
import pathlib, re
import cv2, numpy as np

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
NAMES = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

def cells(mid):
    img = cv2.aruco.drawMarker(DICT, mid, 6)            # 6x6 px, 1 px per cell
    return img > 127                                     # True = white

def verify(mid):
    big = cv2.aruco.drawMarker(DICT, mid, 300)
    big = cv2.copyMakeBorder(big, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    _, ids, _ = cv2.aruco.detectMarkers(big, DICT)
    assert ids is not None and ids.flatten().tolist() == [mid], (mid, ids)

lines = []
for mid, name in NAMES.items():
    verify(mid)
    w = cells(mid)
    white = [f"{c}/{5 - r}" for r in range(6) for c in range(6) if w[r, c]]
    lines.append(f"  \\def\\aruco{name}{{{','.join(white)}}}% ID {mid}")

block = "\n".join(lines)
here = pathlib.Path(__file__).parent
for fn in ("wall.tex", "setup.tex", "aruco.tex"):
    p = here / fn
    if not p.exists():
        continue
    s = p.read_text()
    s2 = re.sub(r"(% GEN ARUCO BEGIN\n).*?([ \t]*% GEN ARUCO END)",
                lambda m: m.group(1) + block + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s2)
    print(fn, "updated" if s2 != s else "unchanged (no markers?)")
print(block)
