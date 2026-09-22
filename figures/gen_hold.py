#!/usr/bin/env python3
"""Generate the hold geometry used by figures/hold.tex.

The hold is a jug: a body tapering away from the wall, whose top is recessed
(the notch / incut) behind a raised lip, ending on a flat front face where the
two counterbored bolt holes open.  Coordinates are in mm; d is the distance
from the wall, z the height, x the horizontal position along the wall.

Running this script rewrites the blocks between the GEN markers in hold.tex.
"""
import math, re, pathlib
import numpy as np

S = 0.58                                 # uniform scale of the jug design
R_ENV = 30.0                             # envelope radius (cylinder, 60 mm)
D = 72.0 * S                             # protrusion of the front face
TOP = [(d*S, z*S) for d, z in
       [(0, 36), (20, 28), (38, 27), (54, 33), (63, 38), (70, 35), (72, 31)]]
BOT = [(d*S, z*S) for d, z in
       [(0, -34), (20, -32), (42, -24), (60, -14), (70, -8), (72, -6)]]
N_EXP = 2.6                              # super-ellipse exponent of sections
W0, W1 = 45.0 * S, 26.0 * S              # half-width at the wall / front face

def catmull(points, n=24):
    p = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = map(np.array, (p[i-1], p[i], p[i+1], p[i+2]))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t*t
                              + (-p0+3*p1-3*p2+p3)*t**3))
    out.append(np.array(points[-1], float))
    return np.array(out)

top, bot = catmull(TOP), catmull(BOT)
zt = lambda d: np.interp(d, top[:, 0], top[:, 1])
zb = lambda d: np.interp(d, bot[:, 0], bot[:, 1])
w  = lambda d: W0 - (W0 - W1) * (d / D) ** 1.3

def section(d, n_ang=40):
    zc, h, hw = (zt(d) + zb(d)) / 2, (zt(d) - zb(d)) / 2, w(d)
    pts = []
    for k in range(n_ang):
        a = 2 * math.pi * k / n_ang
        c, s = math.cos(a), math.sin(a)
        x = hw * math.copysign(abs(c) ** (2 / N_EXP), c)
        z = zc + h * math.copysign(abs(s) ** (2 / N_EXP), s)
        pts.append((x, -d, z))           # world: y < 0 is out of the wall
    return pts

f = lambda v: f"{v:.2f}".rstrip('0').rstrip('.')

# ---------------------------------------------------------------- iso mesh
V = np.array([0.667, -1.0, 0.590]); V /= np.linalg.norm(V)   # toward viewer
L = np.array([-0.35, -0.55, 0.76]); L /= np.linalg.norm(L)   # light
BASE = np.array([0, 150, 136])

rings = [section(d) for d in np.linspace(0, D, 37)]
r_max = max(math.hypot(p[0], p[2]) for r in rings for p in r)
assert r_max < R_ENV, f"hold does not fit the envelope: r = {r_max:.1f} mm"
print(f"max radius {r_max:.1f} mm, protrusion {D:.1f} mm")
faces = []
for r0, r1 in zip(rings, rings[1:]):
    n = len(r0)
    for k in range(n):
        q = [np.array(p) for p in (r0[k], r0[(k+1) % n], r1[(k+1) % n], r1[k])]
        c = sum(q) / 4
        nrm = np.cross(q[1] - q[0], q[3] - q[0])
        axis_pt = np.array([0, c[1], (zt(-c[1]) + zb(-c[1])) / 2])
        if np.dot(nrm, c - axis_pt) < 0:
            nrm = -nrm
        nrm /= np.linalg.norm(nrm)
        if np.dot(nrm, V) <= 0:
            continue                         # back face
        faces.append((np.dot(c, V), q, nrm))
cap = [np.array(p) for p in rings[-1]]
faces.append((1e9, cap, np.array([0.0, -1.0, 0.0])))
faces.sort(key=lambda t: t[0])

mesh = []
for _, q, nrm in faces:
    lam = max(0.0, float(np.dot(nrm, L)))
    rgb = np.clip(BASE * (0.38 + 0.78 * lam) + 255 * 0.10 * lam ** 6, 0, 255).astype(int)
    col = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"
    path = " -- ".join(f"({f(p[0])},{f(p[1])},{f(p[2])})" for p in q)
    mesh.append(f"    \\filldraw[fill={col},draw={col},line width=0.15pt] {path} -- cycle;")

# ------------------------------------------------------ orthographic views
side = [(p[0], p[1]) for p in top] + [(D, -6*S)] + [(p[0], p[1]) for p in bot[::-1]]
side_path = " -- ".join(f"({f(x)},{f(z)})" for x, z in side) + " -- cycle"

def outline(d):
    return " -- ".join(f"({f(p[0])},{f(p[2])})" for p in section(d, 72)) + " -- cycle"

blocks = {
    "FRONT": [f"    \\def\\holdOuter{{{outline(0)}}}",
              f"    \\def\\holdCap{{{outline(D)}}}"],
    "SIDE":  [f"    \\def\\holdside{{{side_path}}}"],
    "ISO":   mesh,
}

tex = pathlib.Path(__file__).with_name("hold.tex")
src = tex.read_text()
for name, lines in blocks.items():
    src = re.sub(rf"(% GEN {name} BEGIN\n).*?(\s*% GEN {name} END)",
                 lambda m: m.group(1) + "\n".join(lines) + m.group(2), src, flags=re.S)
tex.write_text(src)
print(f"{len(mesh)} mesh faces written")
