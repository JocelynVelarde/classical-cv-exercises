import re
import sys
import numpy as np


def parse(text, name):
    m = re.search(rf'{name}\s*=\s*([-\d.eE]+)', text)
    return float(m.group(1))


def parse_matrix(text, name):
    m = re.search(rf'{name}\s*=\s*\[(.*?)\]', text, re.DOTALL)
    rows = m.group(1).strip().split(';')
    return np.array([[float(x) for x in r.split()] for r in rows], dtype=np.float64)


def main():
    with open(sys.argv[1]) as f:
        text = f.read()

    cam0  = parse_matrix(text, "cam0")
    cam1  = parse_matrix(text, "cam1")
    doffs = parse(text, "doffs")          
    base  = parse(text, "baseline") / 1000.0  
    w     = int(parse(text, "width"))
    h     = int(parse(text, "height"))
    ndisp = int(parse(text, "ndisp"))

    fx = float(cam0[0, 0])
    cx = float(cam0[0, 2])
    cy = float(cam0[1, 2])

    distL = np.zeros(5)
    distR = np.zeros(5)
    R1    = np.eye(3)
    R2    = np.eye(3)

    P1 = np.zeros((3, 4)); P1[:3, :3] = cam0
    P2 = np.zeros((3, 4)); P2[:3, :3] = cam1
    P2[0, 3] = -fx * base

    Q = np.array([
        [1.0, 0.0,    0.0,      -cx],
        [0.0, 1.0,    0.0,      -cy],
        [0.0, 0.0,    0.0,       fx],
        [0.0, 0.0,  1.0/base,  doffs/base],
    ], dtype=np.float64)

    np.savez("calib.npz",
             KL=cam0, distL=distL, KR=cam1, distR=distR,
             R=np.eye(3), T=np.array([-base, 0.0, 0.0]),
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             img_size=np.array([w, h]))

    print(f"Resolucion       : {w} x {h}")
    print(f"Focal fx         : {fx:.2f} px")
    print(f"Baseline         : {base*1000:.2f} mm")
    print(f"doffs            : {doffs:.3f} px")
    print(f"ndisp (sugerido) : {ndisp}")


if __name__ == "__main__":
    main()