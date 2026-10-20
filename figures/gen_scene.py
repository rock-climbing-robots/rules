#!/usr/bin/env python3
"""Put the real robot and a real route on the wall of setup.tex.

scene_n3.npz is exported from the climbing project
(learning/rlclimb/export_rules_scene.py): the robot posed in a planned stance
of route N3, its route holds, the rope anchor, all in the figure frame (units
of 100 mm).  The robot and the holds it touches are rendered together with a
depth buffer (climber_setup.png), so the hold rims hide the ball ends; the
other holds use \\realhold from gen_hold.py.  Rewrites the GEN SCENE blocks of
setup.tex.  Run gen_hold.py first.
"""
import pathlib
import re

import numpy as np
import trimesh

from raster import render

HERE = pathlib.Path(__file__).parent
HOLD_RGB = np.array([46, 170, 80])
P_SET = np.array([[0.5, 0.5, 0], [-0.29, 0.29, 0.57]])      # page cm per 100 mm
VIEW = np.cross(P_SET[0], P_SET[1])
VIEW /= np.linalg.norm(VIEW)

sc = np.load(HERE / "scene_n3.npz")
robot = trimesh.Trimesh(sc["verts"], sc["faces"], process=False)

hold = trimesh.load(HERE / "hold.stl", force="mesh")
if hold.extents.max() < 1.0:
    hold.apply_scale(1000.0)
hv = hold.vertices.copy()
hv[:, 1] -= 26.0
hv[:, 2] -= 14.0
hold = trimesh.Trimesh(hv / 100.0, hold.faces, process=True)
hold = hold.simplify_quadric_decimation(face_count=2500)

near = [tuple(h) for h in sc["holds"]]
far = []

parts, cols = [robot], [sc["rgb"]]
for x, z in near:
    parts.append(hold.copy().apply_translation([x, 0.0, z]))
    cols.append(np.tile(HOLD_RGB, (len(hold.faces), 1)))
scene = trimesh.util.concatenate(parts)
node = render(scene, P_SET, VIEW, HERE / "climber_setup.png", ppc=500,
              face_rgb=np.vstack(cols), ss=2)

ax, ay, az = sc["attach"]
cx, cy, cz = sc["centre"]
holds_block = "\n".join(
    ["  % route N3, seed 0: the holds are in the robot render below"]
    + [f"  \\realhold{{{x:.3f}}}{{{z:.3f}}}" for x, z in far]
    + [f"  \\coordinate (Pattach) at ({ax:.3f},{ay:.3f},{az:.3f});% rope anchor on the trunk",
       f"  \\coordinate (Pbot) at ({cx:.3f},{cy:.3f},{cz:.3f});"])
robot_block = (f"  % MicroClimbingBan in stance {int(sc['config'])} of route N3, with the holds around it\n"
               f"  {node}")

setup = HERE / "setup.tex"
src = setup.read_text()
for name, block in (("SCENE HOLDS", holds_block), ("SCENE ROBOT", robot_block)):
    src, n = re.subn(rf"(% GEN {name} BEGIN\n).*?[ \t]*% GEN {name} END",
                     lambda m: m.group(1) + block + f"\n  % GEN {name} END", src, flags=re.S)
    assert n == 1, f"no GEN {name} block in setup.tex"
setup.write_text(src)
print(f"{len(near)} holds rendered with the robot, {len(far)} as \\realhold")
