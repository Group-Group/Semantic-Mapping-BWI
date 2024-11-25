import numpy as np
import open3d as o3d
from collections import defaultdict
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75):
        self._main: list[PointCloud] = []
        self._scene = defaultdict(list)
        self._eps = eps # tolerance
        # self._unmerged_pointclouds = defaultdict(list)

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
    
    def nearest_pointcloud(self, pcl: PointCloud) -> PointCloud:
        nearest_match_dist = float('inf')
        nearest_match = None

        for target in self._scene[pcl.label]:
            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and nearest_match_dist > distance:
                nearest_match_dist = distance
                nearest_match = target

        return nearest_match
    
    def _register_new_pointcloud(self, pcl: PointCloud):
        if not pcl.is_empty():
            self._scene[pcl.label].append(pcl)

    def aggregate_pointcloud(self, pcl: PointCloud, target: PointCloud, transformation: np.ndarray, fitness_rejection: float = 0, verbose: bool = True):
        if not target:
            self._register_new_pointcloud(pcl)
            return
                
        # this is an iterative closest point (icp) algorithm
        # it refines the initial transformation matrix to make the pointclouds match up
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl._pcl, target._pcl, self._eps, transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        if verbose:
            print(f"[ICP] Fitness (higher is better): {reg_p2p.fitness}\tRMSE (lower is better): {reg_p2p.inlier_rmse}")
        
        if reg_p2p.fitness < fitness_rejection:
            print(f"[ICP] fitness rejected ({reg_p2p.fitness} < {fitness_rejection}) new registration detected")
            self._register_new_pointcloud(pcl)
        else:
            # todo determine which pointcloud to merge onto
            pcl = pcl.transform(reg_p2p.transformation)
            target += pcl

        return target
    
    ## ============== Do not use ==============
    # def aggregate_all(self):
    #     for label in self._unmerged_pointclouds:
    #         for instances in self._unmerged_pointclouds[label]:
    #             if len(instances) > 0: # assume pointclouds in this list reference one object
    #                 merged_pcl = self._merge(instances)
    #                 self._scene[label].append(merged_pcl)

    # def _nearest_pointcloud(self, pcl, unmerged_list: list[PointCloud]):
    #     nearest_match_dist = float('inf')
    #     nearest_match_idx = None

    #     for i, target in enumerate(unmerged_list):
    #         distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
    #         distance = np.asarray(distance).mean()
    #         if nearest_match_dist > distance:
    #             nearest_match_dist = distance
    #             nearest_match_idx = i

    #     return unmerged_list[nearest_match_idx]
    
    # def _choose_target(self, pcl1: PointCloud, pcl2: PointCloud, merged_list: list[PointCloud]):
    #     distance1 = 0
    #     distance2 = 0

    #     # maybe dont need to recalculate each time
    #     for target in merged_list:
    #         distance1 += np.asarray(pcl1._pcl.compute_point_cloud_distance(target._pcl)).mean()
    #         distance2 += np.asarray(pcl2._pcl.compute_point_cloud_distance(target._pcl)).mean()

    #     target = pcl1 if min(distance1, distance2) == distance1 else pcl2
    #     source = pcl1 if target == pcl2 else pcl2

    #     return target, source    

    # def _merge(self, pcl_list: list[PointCloud]) -> PointCloud:
    #     while len(pcl_list) > 1:
    #         print("Pointclouds left to merge:", len(pcl_list))
    #         print("==============================")
    #         unmerged_list = pcl_list
    #         merged_list = []

    #         while len(unmerged_list) > 1:
    #             print('cycle:', len(unmerged_list))
    #             # find its nearest neighbor
    #             pcl = unmerged_list.pop()
    #             neighbor = self._nearest_pointcloud(pcl, unmerged_list)
    #             unmerged_list.remove(neighbor)

    #             # minimize the distance between merged pointclouds
    #             target, source = self._choose_target(pcl, neighbor, merged_list)

    #             print('target', target.world_transformation)
    #             print('source', source.world_transformation)

    #             # icp onto target
    #             reg_p2p = o3d.pipelines.registration.registration_icp(
    #                 source._pcl, target._pcl, self._eps * 5, source.world_transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #             )

    #             # complete merge
    #             source.transform(reg_p2p.transformation)
    #             target += source
    #             merged_list += [target]

    #             print("after :))))))))))")
    #             print('target', target.world_transformation)
    #             print('source', source.world_transformation)

    #         pcl_list = merged_list
        
    #     return pcl_list[0]

    # def add_unmerged_pointcloud(self, pcl: PointCloud):
    #     assert len(pcl.world_transformation) > 0
    #     if pcl.points.size > 0:
    #         self._unmerged_pointclouds[pcl.label].append(pcl)

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
            


