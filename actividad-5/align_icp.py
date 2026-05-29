import numpy as np
import open3d as o3d
import copy

VOXEL_SIZE        = 0.01   
NORMAL_RADIUS     = 0.05   
ICP_THRESHOLD     = 0.20   
ICP_MAX_ITER      = 100

pcd1 = o3d.io.read_point_cloud("vista1.ply")
pcd2 = o3d.io.read_point_cloud("vista2.ply")
print(f"vista1.ply: {len(pcd1.points)} points")
print(f"vista2.ply: {len(pcd2.points)} points")

d1 = pcd1.voxel_down_sample(VOXEL_SIZE)
d2 = pcd2.voxel_down_sample(VOXEL_SIZE)
print(f"\nAfter downsampling: {len(d1.points)} / {len(d2.points)} points")

normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30)
d1.estimate_normals(normal_param)
d2.estimate_normals(normal_param)

init = np.eye(4)
reg = o3d.pipelines.registration.registration_icp(
    d2, d1,
    ICP_THRESHOLD,
    init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
)

print(f"\nICP fitness     : {reg.fitness:.4f} ")
print(f"ICP inlier RMSE : {reg.inlier_rmse:.6f} m")
print(reg.transformation)

pcd2_aligned = copy.deepcopy(pcd2)
pcd2_aligned.transform(reg.transformation)

merged = pcd1 + pcd2_aligned
o3d.io.write_point_cloud("reconstruccion.ply", merged)
print(f"\nMerged cloud      : {len(merged.points)} points")
print("Saved             : reconstruccion.ply")

pcd1_c = copy.deepcopy(pcd1); pcd1_c.paint_uniform_color([1.0, 0.5, 0.0])
pcd2_c = copy.deepcopy(pcd2_aligned); pcd2_c.paint_uniform_color([0.0, 0.5, 1.0])
o3d.io.write_point_cloud("reconstruccion_overlay.ply", pcd1_c + pcd2_c)