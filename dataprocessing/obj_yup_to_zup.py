"""Rotate a Y-up .obj into a Z-up .obj (in place on the text, groups preserved).

Every mesh in ``datasets/3dshape`` is extruded along its own **+Y**: the shapes are
10 units thick in y, ``2denv4.obj`` is 350 x 30 x 350, ``env1.obj`` is 199 x 140 x 210.
``preprocess_obj.py --2d``, however, defines the planar slice as **z = 0** -- it flattens
``V_env[:, 2]`` and squashes the shape's z extent.  Pointing ``--2d`` at a Y-up mesh
therefore collapses one of the two *in-plane* axes: the 350 x 350 maze becomes a
350 x 30 side silhouette that fills the whole sampling box, no placement is ever
collision-free, and the rejection loop in ``generate_valid_pairs`` spins forever.

This script applies R_x(90 deg) -- ``(x, y, z) -> (x, -z, y)`` -- so the extrusion axis
becomes +Z and ``--2d`` sees the geometry it expects.  Only ``v``/``vn`` lines are
rewritten, so groups, materials and face indices survive untouched.

    python dataprocessing/obj_yup_to_zup.py datasets/3dshape/2denv4.obj
    python dataprocessing/obj_yup_to_zup.py datasets/3dshape/Tshape3d.obj
    # -> datasets/3dshape/2denv4_zup.obj, datasets/3dshape/Tshape3d_zup.obj
"""

import argparse
import os


def rotate_line(prefix, line):
    """Rewrite one 'v'/'vn' line: (x, y, z) -> (x, -z, y)."""
    parts = line.split()
    x, y, z = (float(v) for v in parts[1:4])
    rest = parts[4:]
    out = "{} {:.6f} {:.6f} {:.6f}".format(prefix, x, -z, y)
    return (out + " " + " ".join(rest) if rest else out) + "\n"


def convert(src, dst):
    with open(src) as f:
        lines = f.readlines()
    out = []
    n = 0
    for line in lines:
        head = line.split(" ", 1)[0]
        if head in ("v", "vn") and len(line.split()) >= 4:
            out.append(rotate_line(head, line))
            n += 1
        else:
            out.append(line)
    with open(dst, "w") as f:
        f.writelines(out)
    print("{} -> {}  ({} v/vn lines rotated)".format(src, dst, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help=".obj file(s) to rotate")
    ap.add_argument("--suffix", default="_zup")
    ap.add_argument("--out", default=None, help="explicit output path (single input only)")
    args = ap.parse_args()

    if args.out is not None:
        assert len(args.src) == 1, "--out takes a single input"
        convert(args.src[0], args.out)
        return
    for src in args.src:
        root, ext = os.path.splitext(src)
        convert(src, root + args.suffix + ext)


if __name__ == "__main__":
    main()
