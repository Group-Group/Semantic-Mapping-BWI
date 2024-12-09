import numpy as np
import open3d as o3d
from collections import defaultdict
from pointcloud import PointCloud

class PointCloudAggregator:
    def __init__(self, eps=0.75, icp_eps=0.05):
        self._main: PointCloud = None
        self._scene = defaultdict(list)
        self._eps = eps # tolerance
        self._icp_eps = icp_eps
        self._unmerged_pointclouds = defaultdict(list)

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        # compile main pointcloud and return
        self._main = PointCloud()
        for label in self._scene:
            for pcl in self._scene[label]:
                if len(pcl) > 100:
                    self._main += pcl
                # else:
                #     self._remove_pointcloud(pcl)

        return self._main
    
    def nearest_pointcloud(self, pcl: PointCloud) -> PointCloud:
        nearest_match_dist = float('inf')
        nearest_match = None

        for target in self._scene[pcl.label]:
            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and distance < nearest_match_dist:
                nearest_match_dist = distance
                nearest_match = target

        return nearest_match
    
    def _register_pointcloud(self, pcl: PointCloud):
        if not pcl.is_empty():
            self._scene[pcl.label].append(pcl)

    def _remove_pointcloud(self, pcl: PointCloud):
        self._scene[pcl.label].remove(pcl)

    def aggregate_pointcloud(self, pcl: PointCloud, target: PointCloud, verbose: bool = True):
        if not target:
            self._register_pointcloud(pcl)
            return
        
        ## todo determine criteria for switching target and source
        ## todo target should not have incomplete data (so it will be the most complete pointcloud we have)
        # criterion = lambda x: x._pcl.get_axis_alinged_bounding_box().volume() / len(x)
        
        # if criterion(pcl) > criterion(target):
        #     target, pcl = pcl, target
        #     remove target from dictionary, and replace it with source

        # icp to refine initial transformation
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl._pcl, target._pcl, self._icp_eps, np.eye(4), 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
        )
        if verbose:
            print(f"[ICP] Fitness (higher is better): {reg_p2p.fitness}\tRMSE (lower is better): {reg_p2p.inlier_rmse}")
        
        self._remove_pointcloud(target)
        pcl = pcl.transform(reg_p2p.transformation)
        target += pcl
        self._register_pointcloud(target)

        return target
    
    def gather_pointclouds(self):
        """
        starting state: scene['chair'] -> list of all chairs in the scene (untransformed)
        ending state: scene['chair'] -> list of all chairs in the scene placed into groups (untransformed)

        testing 12/5
            - maintain untransformed pointclouds (always, makes it easier to do icp)
                >>> icp(source, target) -> new transformation for source
                >>> target + source.transform(icp_transformation) -> merged pointcloud
                
                maybe keep this in a list
                once algo is done, apply all transforms and + into one pointcloud
                
            
            why is it necessary to choose source and target? some frames have more complete data, these should be the target
        """
        pass

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

class PointCloudAggregator2:
    def __init__(self, eps=0.75, icp_eps=0.05):
        self._main: PointCloud = None
        self._scene = defaultdict(list) # label -> list[list[PointCloud]]
        self._eps = eps # tolerance
        self._icp_eps = icp_eps

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        # compile main pointcloud and return
        self._main = PointCloud()
        for label in self._scene:
            for pcl in self._scene[label]:
                if len(pcl) > 100:
                    self._main += pcl
                # else:
                #     self._remove_pointcloud(pcl)

        return self._main
    

    def _register_pointcloud(self, pcl):
        self._scene[pcl.label] += [[pcl]]

    def nearest_pointcloud(self, pcl):
        nearest_match_dist = float('inf')
        nearest_match = None

        for group in self._scene[pcl.label]:
            target = PointCloud()
            for instance in group:
                target += instance

            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and distance < nearest_match_dist:
                nearest_match_dist = distance
                nearest_match = group

        return nearest_match
    
    def aggregate_pointcloud(self, pcl, target_group, min_view_threshold=20):
        if not target_group: # make a new target group
            self._register_pointcloud(pcl)
            return
        
        # find target
        nearest_match_dist = float('inf')
        min_rot_dist = float('inf')
        rotation = pcl.world_transformation[:3, :3]
        target = None
        for instance in target_group:
            distance = pcl._pcl.compute_point_cloud_distance(instance._pcl)
            distance = np.asarray(distance).mean()
            if distance < nearest_match_dist:
                nearest_match_dist = distance
                target = instance
            
            r = instance.world_transformation[:3, :3]
            rot_deg = self.calculate_rotation_angle(rotation, r)
            min_rot_dist = min(min_rot_dist, rot_deg)

        if min_rot_dist > min_view_threshold: # it's a new view of the object (gain at least 30 deg)
            target_group.append(pcl)
            return

        if pcl.score() > target.score(): # found a better representation of the object
            target_group.remove(target)
            target_group.append(pcl)

    # def aggregate_all(self, verbose=False):
    #     new_scene = defaultdict(list)

    #     for label in self._scene:
    #         for group in self._scene[label]:
    #             # merge pointclouds whose rotations are close to each other, don't touch the ones that aren't
    #             new_group = []

    #             for instance in group:
    #                 rotation = instance.world_transformation[:3, :3]
    #                 group.remove(instance)
    #                 for target in group:
    #                     r = target.world_transformation[:3, :3]
    #                     rot_deg = self.calculate_rotation_angle(rotation, r)
                        
    #                     if rot_deg < 20: # close enough in rotation to do icp
    #                         reg_p2p = o3d.pipelines.registration.registration_icp(
    #                         instance._pcl, target._pcl, self._icp_eps, np.eye(4), 
    #                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    #                     )
                            
    #                         if verbose:
    #                             print(f"[ICP] Fitness (higher is better): {reg_p2p.fitness}\tRMSE (lower is better): {reg_p2p.inlier_rmse}")
                        
    #                         instance.transform(reg_p2p.transformation)

    #                 new_group.append(instance)
                
    #             grouped_pcl = PointCloud()
    #             grouped_pcl.label = label
    #             for pcl in new_group:
    #                 grouped_pcl += pcl

    #             new_scene[label] += [grouped_pcl]
        
    #     self._scene = new_scene


    def calculate_rotation_angle(self, R1, R2):
        R_rel = np.dot(R1.T, R2)  # R1^T * R2
        trace_R_rel = np.trace(R_rel)
        cos_theta = (trace_R_rel - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
