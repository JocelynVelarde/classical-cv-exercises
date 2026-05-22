import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
# Prohibited to use pyrealsense2, kornia, torch, opencv-contrib-python (cv2.rgbd.depthTo3d function), PyntCloud, o3d or Pillow libraries

# 1. Load the K matrix from the CSV file
K = []
with open("camera_intrinsics.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
            K.append([float(v) for v in row])

K = np.array(K, dtype=np.float64)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"Intrínsecos -> fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

# 2.a Load the aligned color image
color_image = cv2.imread("aligned_color.png")
color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# 2.b Read the raw 16-bit depth
depth_image = cv2.imread("aligned_depth_raw.png", cv2.IMREAD_UNCHANGED)

# 3. 3D Coordinate Calculation
height, width = depth_image.shape
depth_scale = 0.001
Z = depth_image.astype(np.float64) * depth_scale
u = np.arange(width)
v = np.arange(height)
uu, vv = np.meshgrid(u, v)
X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
colors = color_rgb.reshape(-1, 3)

valid = Z.reshape(-1) > 0
points = points[valid]
colors = colors[valid]

# 4. Display the 3D result
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

step = max(1, len(points) // 30000)
pts_s = points[::step]
col_s = colors[::step] / 255.0 

ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
           c=col_s, s=1, marker=".")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Pointcloud 3D")

ax.invert_yaxis()
plt.tight_layout()
plt.savefig("pointcloud.png", dpi=120)
plt.show()

# 5. Export to PLY format (for MeshLab)
def write_ply(filename, points, colors):
    assert points.shape[0] == colors.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n")

write_ply("scene.ply", points, colors)