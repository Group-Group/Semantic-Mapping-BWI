from pointcloud import PointCloud
import numpy as np
import open3d as o3d

class PointCloudView:
    """
    A view of an object from a specific range of angles
    """
    def __init__(self, pointclouds: PointCloud, theta: float):
        if not isinstance(pointclouds, list):
            pointclouds = [pointclouds]
        self.pointclouds: list[PointCloud] = pointclouds
        self._theta: float = theta

    def get_min_angle_gain(self, R: np.ndarray):
        # calculate the minimum gain (in degrees) of adding this view
        min_gain = float('inf')
        for pcl in self.pointclouds:
            min_gain = min(min_gain, self._calculate_rotation_angle(pcl.rotation, R))
        return min_gain

    def is_same_view(self, R: np.ndarray):
        # check if a rotation array represents the same view
        delta_degrees = self.get_min_angle_gain(R)
        return delta_degrees <= self._theta
    
    def get_pointcloud(self):
        # Return the pointcloud represented by this view
        pcl = sum(self.pointclouds, PointCloud())
        return pcl
    
    def refine_view(self, eps, max_iters):
        # icp all pointclouds in this view
        assert len(self.pointclouds) > 0

        # pick target, the most complete pointcloud
        target = self.pointclouds[0]
        for pcl in self.pointclouds:
            if pcl.score() > target.score():
                target = pcl
        
        # icp all pointclouds onto target
        self.pointclouds.remove(target)
        for source in self.pointclouds:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source._pcl, target._pcl, eps, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)
            )

            source.transform(reg_p2p.transformation)
            target += pcl
        
        self.pointclouds = [target]

    def remove(self, pcl):
        self.pointclouds.remove(pcl)

    def add(self, pcl):
        self.pointclouds.append(pcl)

    def _calculate_rotation_angle(self, R1: np.ndarray, R2: np.ndarray):
        R_rel = np.dot(R1.T, R2)
        trace_R_rel = np.trace(R_rel)
        cos_theta = (trace_R_rel - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees # in [0, 180]
    
    def __add__(self, view):
        self.pointclouds += view.pointclouds
        return self