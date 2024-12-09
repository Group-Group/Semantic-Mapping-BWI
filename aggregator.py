import numpy as np
from collections import defaultdict
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75):
        self._main: PointCloud = None
        self._scene = defaultdict(list) # label -> list[list[PointCloud]]
        self._eps = eps # tolerance

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        self._flatten_scene()

        # compile main pointcloud and return
        self._main = PointCloud()
        for label in self._scene:
            for pcl in self._scene[label]:
                if len(pcl) > 100:
                    self._main += pcl
                # else:
                #     self._remove_pointcloud(pcl)

        return self._main
    
    def _flatten_scene(self):
        new_scene = defaultdict(list)
        for label in self._scene:
            for group in self._scene[label]:
                if not isinstance(group[0], list):
                    return
                pcl = sum(group, PointCloud(label=label))
                new_scene[label].append(pcl)
        self._scene = new_scene

    def _register_pointcloud(self, pcl: PointCloud):
        self._scene[pcl.label] += [[pcl]]

    def nearest_pointcloud(self, pcl: PointCloud):
        nearest_match_dist = float('inf')
        nearest_match = None

        for group in self._scene[pcl.label]:
            target = sum(group, PointCloud())
            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance < nearest_match_dist <= self._eps:
                nearest_match_dist = distance
                nearest_match = group

        return nearest_match
    
    def aggregate_pointcloud(self, pcl: PointCloud, target_group: list[PointCloud], min_view_threshold=20):
        if not target_group: # make a new target group
            self._register_pointcloud(pcl)
            return
        
        # find target
        nearest_match_dist = float('inf')
        min_rot_dist = float('inf')
        target = None
        for instance in target_group:
            distance = pcl._pcl.compute_point_cloud_distance(instance._pcl)
            distance = np.asarray(distance).mean()
            if distance < nearest_match_dist:
                nearest_match_dist = distance
                target = instance
            
            rot_deg = self.calculate_rotation_angle(pcl.rotation, instance.rotation)
            min_rot_dist = min(min_rot_dist, rot_deg)

        if min_rot_dist > min_view_threshold: # it's a new view of the object (gain at least 30 deg)
            target_group.append(pcl)
            return

        if pcl.score() > target.score(): # found a better representation of the object
            target_group.remove(target)
            target_group.append(pcl)

    def calculate_rotation_angle(self, R1, R2):
        R_rel = np.dot(R1.T, R2)  # R1^T * R2
        trace_R_rel = np.trace(R_rel)
        cos_theta = (trace_R_rel - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
