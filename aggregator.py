import numpy as np
from collections import defaultdict
from pointcloud import PointCloud
from views import PointCloudView

class PointCloudAggregator:
    def __init__(self, eps:float=0.75):
        self._main: PointCloud = None
        self._scene: defaultdict[str, list[list[PointCloudView]]] = defaultdict(list)
        self._eps = eps # tolerance

    @property
    def scene(self):
        return self._scene
    
    @scene.getter
    def scene(self):
        return self._flatten_scene()

    @property
    def main(self):
        return self._main
    
    @main.getter
    def main(self):
        # compile main pointcloud and return
        self._main = PointCloud()
        for label in self.scene:
            for pcl in self.scene[label]:
                if len(pcl) > 100:
                    self._main += pcl

        return self._main
      
    def nearest_pointcloud(self, pcl: PointCloud) -> list[PointCloudView]:
        nearest_match_dist = float('inf')
        nearest_match = None

        for instance in self._scene[pcl.label]:
            target = PointCloud()
            for view in instance:
                target += view.get_pointcloud()

            distance = pcl._pcl.compute_point_cloud_distance(target._pcl)
            distance = np.asarray(distance).mean()
            if distance <= self._eps and distance < nearest_match_dist:
                nearest_match_dist = distance
                nearest_match = instance

        return nearest_match

    def aggregate_pointcloud(self, pcl: PointCloud, instance: list[PointCloudView], min_view_threshold=20):
        if not instance: # its a new object
            self._register_pointcloud(pcl, min_view_threshold)
            return
        
        # find target
        nearest_match_dist = float('inf')
        min_rot_dist = float('inf')
        target = None
        target_view = None
        for view in instance:
            other = view.get_pointcloud()
            distance = pcl._pcl.compute_point_cloud_distance(other._pcl)
            distance = np.asarray(distance).mean()
            if distance < nearest_match_dist:
                nearest_match_dist = distance
                target = other
                target_view = view
            
            min_rot_dist = min(min_rot_dist, view.get_min_angle_gain(other.rotation))

        if min_rot_dist > min_view_threshold: # it's a new view of the object (gain at least 20 deg)
            instance.append(PointCloudView(pcl, min_view_threshold))
            return

        if pcl.score() > target.score(): # found a better representation of the object
            target_view.remove(target)
            target_view.add(pcl)

    def refine_views(self, eps=0.50, max_iters=500):
        # do icp to refine views
        # carson said icp worked for him, so this is my last attempt
        for label in self._scene:
            for instance in self._scene[label]:
                for view in instance:
                    view.refine_view(eps, max_iters)

    def _flatten_scene(self):
        new_scene = defaultdict(list)
        for label in self._scene:
            for instance in self._scene[label]:
                pcl = PointCloud(label=label)
                for view in instance:
                    pcl += view.get_pointcloud()

                new_scene[label].append(pcl)
        return new_scene

    def _register_pointcloud(self, pcl: PointCloud, min_view_threshold: float):
        self._scene[pcl.label] += [[PointCloudView(pcl, min_view_threshold)]]
