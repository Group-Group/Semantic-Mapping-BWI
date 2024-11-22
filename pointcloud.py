from datetime import datetime
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

class PointCloud:
    def __init__(self, points=[], label=None):
        points = [] if isinstance(points, np.ndarray) and points.shape[0] == 0 else points
        self._pcl = o3d.geometry.PointCloud()
        self._pcl.points = o3d.utility.Vector3dVector(points)
        self.label = label
        self.timestamp = datetime.now()
    
    @property
    def points(self):
        return np.asarray(self._pcl.points)
    
    @points.setter
    def points(self, value):
        self._pcl = o3d.geometry.PointCloud()
        self._pcl.points = o3d.utility.Vector3dVector(value)
        self.timestamp = datetime.now()

    def is_empty(self):
        return len(self._pcl.points) == 0

    def transform(self, transformation):
        # in place
        self._pcl.transform(transformation)
        return self
    
    def clean(self, eps=0.05, min_points=10, verbose: bool=True):
        # deduplication
        self._pcl.remove_duplicated_points()

        # remove outliers with dbscan
        labels = self._pcl.cluster_dbscan(eps=eps, min_points=min_points)
        labels = np.array(labels)
        noise = labels == -1
        if verbose:
            print(f"[DBSCAN] Found {labels.max() + 1} clusters")
            print(f"[DBSCAN] Removing {noise.sum()} noise points")
        self.points = self.points[~noise]
        return self
    
    def save(self, filename=None):
        filename = filename or str(self.timestamp).replace(" ", "_") + "_" + self.label
        vertex = np.array([(x, y, z) for x, y, z in self.points], 
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(filename)

    def __getitem__(self, indices):
        return self._points[indices]

    def __str__(self):
        return str(self.points)
    
    def __add__(self, pcl):
        combined = np.vstack((self.points, pcl.points))
        return PointCloud(combined, self.label)
    