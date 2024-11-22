import numpy as np
import open3d as o3d
from collections import defaultdict
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75):
        self._main = []
        self._scene = defaultdict(list)
        self._eps = eps # tolerance
        self._unmerged_pointclouds = defaultdict(list) # label -> list[(PointCloud, transform)]

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        # compile main pointcloud and return
        self._main = []
        for obj in self._scene:
            for pcl in self._scene[obj]:
                self._main += [pcl]

        return self._main
    
    def nearest_pointcloud(self, pcl: PointCloud) -> PointCloud:
        # old method: centroid matching (bad) switched to nearest point matching with average
        nearest_match_dist = float('inf')
        nearest_match = None

        for target in self._scene[pcl.label]:
            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and nearest_match_dist > distance:
                nearest_match_dist = distance
                nearest_match = target

        return nearest_match
    
    def aggregate_pointcloud(self, pcl: PointCloud, target: PointCloud, transformation: np.ndarray, verbose: bool = True):
        if not target: # new pointcloud
            if not pcl.is_empty():
                self._scene[pcl.label].append(pcl)
            return
        
        # this is an iterative closest point (icp) algorithm
        # it refines the initial transformation matrix to make the pointclouds match up
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl._pcl, target._pcl, self._eps, transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        if verbose:
            print(f"[ICP] Fitness (higher is better): {reg_p2p.fitness}\tRMSE (lower is better): {reg_p2p.inlier_rmse}")
        pcl = pcl.transform(reg_p2p.transformation)
        target += pcl
        return target
    
    def aggregate_all(self):
        def nearest_pointcloud(pcl, unmerged_list: list[tuple[PointCloud, np.ndarray]]):
            nearest_match_dist = float('inf')
            nearest_match_idx = None

            for i, (target, transform) in enumerate(unmerged_list):
                distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
                distance = np.asarray(distance).mean()
                if nearest_match_dist > distance:
                    nearest_match_dist = distance
                    nearest_match_idx = i

            return unmerged_list[nearest_match_idx], nearest_match_idx
        
        def choose_target(pcl1: PointCloud, pcl2: PointCloud, merged_list: list[PointCloud]):
            distance1 = 0
            distance2 = 0

            # maybe dont need to recalculate each time
            for target in merged_list:
                distance1 += pcl1._pcl.compute_point_cloud_distance(target._pcl)
                distance2 += pcl2._pcl.compute_point_cloud_distance(target._pcl)

            target = pcl1 if min(distance1, distance2) == distance1 else pcl2
            source = pcl1 if target == pcl2 else pcl2

            return target, source
        
        def merge(pcl_list: list[tuple[PointCloud, np.ndarray]]):
            while len(pcl_list) > 1:
                unmerged_mask = np.ones_like(pcl_list, dtype=bool)
                merged_list = []
                new_transforms = []

                while len(pcl_list[unmerged_mask]) > 1:
                    # find its nearest neighbor
                    pcl, my_transformation = pcl_list[unmerged_mask][0]
                    unmerged_mask[0] = False
                    (neighbor, other_transformation), idx = nearest_pointcloud(pcl, pcl_list[unmerged_mask])

                    # minimize the distance between merged pointclouds
                    target, source = choose_target(pcl, neighbor, merged_list)

                    # icp onto target
                    unmerged_mask[0] = True
                    transformation = my_transformation if target == pcl else other_transformation
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        source, target, self._eps, transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    )

                    # complete merge
                    source.transform(reg_p2p.transformation)
                    target += source
                    merged_list += [target]
                    new_transforms += [transformation]
                    unmerged_mask[[0, idx]] = False

                pcl_list = [(merged_pcl, transformation) for merged_pcl, transformation in zip(merged_list, new_transforms)]
            
            return pcl_list[0]

        for label in self._unmerged_pointclouds:
            for instances in self._unmerged_pointclouds[label]:
                if len(instances) > 0: # assume pointclouds in this list reference one object
                    merged_pcl = merge(instances)
                    self._scene[label].append(merged_pcl)

    def add_unmerged_pointcloud(self, pcl: PointCloud, transform: np.ndarray):
        self._unmerged_pointclouds[pcl.label].append((PointCloud, transform))