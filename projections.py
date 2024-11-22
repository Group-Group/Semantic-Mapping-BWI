import numpy as np
import open3d as o3d
import cv2
from pointcloud import PointCloud

## NFOV
cx = 638.4906616210938
cy = 364.21429443359375
fx = 614.5958251953125
fy = 614.3775634765625

## WFOV
# cx = 957.9860229492188
# cy = 546.5714111328125
# fx = 921.8936767578125
# fy = 921.5663452148438

## Distortion coefficients
k1 = 0.4219205677509308
k2 = -2.3591108322143555
k3 = 1.3102449178695679
k4 = 0.3047582805156708
k5 = -2.1945600509643555
k6 = 1.2463057041168213

p2 = -0.0004227574390824884
p1 = 0.0006175598246045411


class PointProjector:
    def __init__(self):
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.eye(4)
        K_inv[:3, :3] = self._K
        K_inv = np.linalg.inv(K_inv) ## 2d pixel coords to camera coords
        self._K_inv = K_inv
        self._D = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        self._map1, self._map2 = cv2.initUndistortRectifyMap(self._K, self._D, np.array([]), self._K, (1280, 720), cv2.CV_32FC1)

    def get_pointcloud(self, depth_image: np.ndarray, stride: int = 3) -> PointCloud:
        rows, cols = depth_image.shape
        points = []

        for u in range(0, cols, stride):
            for v in range(0, rows, stride):
                depth_value = depth_image[v, u] / 1000
                if depth_value == 0.0:
                    continue

                uv_h = np.array([u, v, 1., 1 / depth_value])
                point = depth_value * (self._K_inv @ uv_h)
                if (np.isnan(point).any()):
                    continue

                points.append(point[:3])
        
        points = np.array(points)
        return PointCloud(points)

    def undistort_image(self, image):
        return np.array(cv2.remap(image, self._map1, self._map2, cv2.INTER_NEAREST))
    
    def visualize(self, pcl: PointCloud | list[PointCloud]):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PointCloud", width=800, height=800)

        def add_pointcloud(cloud):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud.points)
            vis.add_geometry(pcd)

        if not isinstance(pcl, list):
            pcl = [pcl]
        for cloud in pcl:
            add_pointcloud(cloud)
        
        try:
            vis.run()
        finally:
            vis.destroy_window()
            