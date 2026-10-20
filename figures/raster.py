"""Depth-buffered rendering of a triangle mesh to a transparent PNG.

Shared by gen_hold.py and gen_scene.py.  Painting triangles far-to-near (what
a vector figure can do) gets concave parts wrong; a depth buffer does not.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHT = np.array([-0.35, -0.55, 0.76])
LIGHT /= np.linalg.norm(LIGHT)


def render(mesh, P, view, path, ppc, face_rgb=None, ss=3):
    """Render `mesh` (trimesh) through the 2x3 page projection P (page cm per
    mesh unit), seen along `view` (pointing to the viewer).  face_rgb: (F, 3)
    base colours 0-255, or one colour.  Curved surfaces are shaded smoothly,
    sharp edges stay flat.  Writes `path` and returns the TikZ node placing it
    (cm, relative to the current origin)."""
    vtx = np.asarray(mesh.vertices, float)
    fac = np.asarray(mesh.faces)
    fn = np.asarray(mesh.face_normals, float)
    vn = np.asarray(mesh.vertex_normals, float)
    base = np.broadcast_to(np.asarray(face_rgb, float), (len(fac), 3))
    sm = vn[fac].mean(axis=1)
    sm /= np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-12)
    sharp = (sm * fn).sum(axis=1) < np.cos(np.radians(12))
    s2 = vtx @ P.T
    dep = vtx @ view
    lo2, hi2 = s2.min(axis=0), s2.max(axis=0)
    k = ppc * ss
    W = (int(np.ceil((hi2[0] - lo2[0]) * k)) + 2 + ss) // ss * ss
    H = (int(np.ceil((hi2[1] - lo2[1]) * k)) + 2 + ss) // ss * ss
    px = (s2[:, 0] - lo2[0]) * k + 1
    py = (hi2[1] - s2[:, 1]) * k + 1
    zb = np.full((H, W), -np.inf, np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    for t in np.nonzero(fn @ view > 0)[0]:
        i = fac[t]
        xs, ys = px[i], py[i]
        x0, x1 = max(int(xs.min()), 0), min(int(np.ceil(xs.max())), W - 1)
        y0, y1 = max(int(ys.min()), 0), min(int(np.ceil(ys.max())), H - 1)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if x1 < x0 or y1 < y0 or abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        z = w0 * dep[i[0]] + w1 * dep[i[1]] + w2 * dep[i[2]]
        win = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        upd = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (z > zb[win])
        if not upd.any():
            continue
        if sharp[t]:
            nrm = np.broadcast_to(fn[t], w0.shape + (3,))
        else:
            nrm = w0[..., None] * vn[i[0]] + w1[..., None] * vn[i[1]] + w2[..., None] * vn[i[2]]
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        lam = np.clip(nrm @ LIGHT, 0, 1)[..., None]
        col = base[t] * (0.38 + 0.78 * lam) + 25.5 * lam ** 6
        zb[win][upd] = z[upd]
        rgb[win][upd] = col[upd]
        alpha[win][upd] = 1.0
    blk = lambda a: a.reshape(H // ss, ss, W // ss, ss, *a.shape[2:])
    asum = blk(alpha).sum(axis=(1, 3))
    c = blk(rgb * alpha[..., None]).sum(axis=(1, 3)) / np.maximum(asum, 1e-9)[..., None]
    img = np.dstack([np.clip(c / 255.0, 0, 1), asum / ss ** 2])
    plt.imsave(path, img)
    left, bottom = lo2[0] - 1 / k, hi2[1] - (H - 1) / k
    return (f"\\node[anchor=south west,inner sep=0pt] at ({left:.4f}cm,{bottom:.4f}cm) "
            f"{{\\includegraphics[width={W / k:.4f}cm]{{{path.name}}}}};")
