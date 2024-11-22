import numpy as np
import open3d as o3d
from collections import defaultdict
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75):
        self._main = []
        self._scene = defaultdict(list)
        self._eps = eps # tolerance

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

    def merge_pointclouds(self, pcl_list: np.ndarray[PointCloud], transformations: np.ndarray[np.ndarray]) -> PointCloud:

        def nearest_pointcloud(pcl, unmerged_list):
            nearest_match_dist = float('inf')
            nearest_match_idx = None
            nearest_match = None

            for i, target in enumerate(unmerged_list):
                distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
                distance = np.asarray(distance).mean()
                if nearest_match_dist > distance:
                    nearest_match_dist = distance
                    nearest_match = target
                    nearest_match_idx = i

            return nearest_match, nearest_match_idx
        
        def choose_target(pcl1, pcl2, merged_list):
            distance1 = 0
            distance2 = 0

            for target in merged_list:
                distance1 += pcl1._pcl.compute_point_cloud_distance(target._pcl)
                distance2 += pcl2._pcl.compute_point_cloud_distance(target._pcl)

            return pcl1 if min(distance1, distance2) == distance1 else pcl2
        
        while len(pcl_list) > 1: # loop until the number of objects in scene is right
            unmerged_mask = np.ones_like(pcl_list, dtype=bool)
            merged_list = []
            new_transforms = []

            while len(pcl_list[unmerged_mask]) > 1: # while unmerged pointclouds
                pcl = pcl_list[unmerged_mask][0]
                neighbor, idx = nearest_pointcloud(pcl, pcl_list[unmerged_mask])
                target = choose_target(pcl, neighbor, merged_list)
                transformation = transformations[unmerged_mask][0] if target == pcl else transformations[unmerged_mask][idx]
                
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcl._pcl, target._pcl, self._eps, transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                pcl = pcl.transform(reg_p2p.transformation)
                target += pcl
                merged_list += [target]
                new_transforms += [reg_p2p.transformation]
                unmerged_mask[[0, idx]] = False
            
            pcl_list = np.array(merged_list)
            transformations = np.array(new_transforms)

        return pcl_list[0]

    
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
        # todo try an iterative aggregator using icp
        # start with small epsilons, go larger with more iterations
        # 1st pass has lots of bad pointclouds
        # 2nd pass has fewer
        # continue until stop criteria
        return target
    
