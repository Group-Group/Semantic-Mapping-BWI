import numpy as np
import open3d as o3d
from collections import defaultdict
from copy import deepcopy
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75):
        self._main: list[PointCloud] = []
        self._scene = defaultdict(list)
        self._eps = eps # tolerance
        self._unmerged_pointclouds = defaultdict(list)

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        # compile main pointcloud and return
        self._main = []
        for label in self._scene:
            for pcl in self._scene[label]:
                self._main += [pcl]

        return self._main
    
    def nearest_pointcloud(self, pcl: PointCloud, transformation: np.ndarray) -> PointCloud:
        copy = deepcopy(pcl)
        copy.transform(transformation)
        nearest_match_dist = float('inf')
        nearest_match = None

        for target in self._scene[copy.label]:
            distance = copy._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and distance < nearest_match_dist:
                nearest_match_dist = distance
                nearest_match = target

        return nearest_match
    
    def _register_new_pointcloud(self, pcl: PointCloud):
        if not pcl.is_empty():
            self._scene[pcl.label].append(pcl)

    def aggregate_pointcloud(self, pcl: PointCloud, target: PointCloud, transformation: np.ndarray, fitness_rejection: float = 0, verbose: bool = True):
        if not target:
            pcl.transform(transformation)
            self._register_new_pointcloud(pcl)
            return
        
        criterion = lambda x: x._pcl.get_axis_alinged_bounding_box().volume() / len(x)
        
        if criterion(pcl) > criterion(target):
            target, pcl = pcl, target

        
        # this is an iterative closest point (icp) algorithm
        # it refines the initial transformation matrix to make the pointclouds match up
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl._pcl, target._pcl, self._eps, transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        if verbose:
            print(f"[ICP] Fitness (higher is better): {reg_p2p.fitness}\tRMSE (lower is better): {reg_p2p.inlier_rmse}")
        
        if reg_p2p.fitness < fitness_rejection:
            print(f"[ICP] fitness rejected ({reg_p2p.fitness} < {fitness_rejection}) new registration detected")
            pcl.transform(transformation)
            self._register_new_pointcloud(pcl)
        else:
            # todo determine which pointcloud to merge onto
            pcl = pcl.transform(reg_p2p.transformation)
            target += pcl

        return target

    # def aggregate_all_pointclouds(self):

    #     for label in self._unmerged_pointclouds:

    #         for group in self._unmerged_pointclouds[label]:

    #             while len(group) > 1:
    #                 p1, p2 = group.pop(), group.pop()

    #                 self.aggregate_pointcloud(p1, p2,)
            

    # def gather_pointclouds(self, eps=None, min_points=2):
    #     eps = eps or self._eps
    #     ## each pointcloud is an observation of an object
    #     ## it needs to be placed in the right bucket before merging
        
    #     gathered_pointclouds = defaultdict(list)
    #     ## try a dbscan approach, reduce each pointcloud to its centroid
    #     for label in self._unmerged_pointclouds:
    #         centroids = []

    #         for target in self._unmerged_pointclouds[label]:
    #             centroids.append(target.points.mean(axis=0)) # (x, y, z)
            
    #         centroids = PointCloud(centroids)
    #         buckets = centroids._pcl.cluster_dbscan(eps=eps, min_points=min_points)
    #         buckets = np.array(buckets)
    #         buckets = buckets[buckets != -1]

    #         observations = np.array(self._unmerged_pointclouds[label])
    #         gathered_pointclouds[label] = [0] * len(np.unique(buckets))
    #         for i, bucket in enumerate(np.unique(buckets)):
    #             gathered_pointclouds[label][i] = observations[buckets == bucket].tolist()

    #     self._unmerged_pointclouds = gathered_pointclouds
            

