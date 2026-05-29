import numpy as np
import open3d as o3d
import copy

SHIFT_X    = 0.15             
ROT_Y_DEG  = 5.0               

pcd1 = o3d.io.read_point_cloud("vista1.ply")
print(f"vista1.ply: {len(pcd1.points)} points")

theta = np.deg2rad(ROT_Y_DEG)
T = np.eye(4)
T[:3, :3] = np.array([
    [ np.cos(theta), 0, np.sin(theta)],
    [ 0,             1, 0            ],
    [-np.sin(theta), 0, np.cos(theta)],
])
T[:3, 3] = [SHIFT_X, 0.0, 0.0]

print(T)

pcd2 = copy.deepcopy(pcd1)
pcd2.transform(T)

o3d.io.write_point_cloud("vista2.ply", pcd2)
print(f"\nvista2.ply saved: {len(pcd2.points)} points")