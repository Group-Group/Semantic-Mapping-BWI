from datetime import datetime
import numpy as np
import open3d as o3d
# from plyfile import PlyData, PlyElement

class PointCloud:
    def __init__(self, points=[], colors=[], label=None, transformation=np.eye(4)):
        points = [] if isinstance(points, np.ndarray) and points.shape[0] == 0 else points
        colors = [] if isinstance(colors, np.ndarray) and colors.shape[0] == 0 else colors
        self._pcl = o3d.geometry.PointCloud()
        self._pcl.points = o3d.utility.Vector3dVector(points)
        self._pcl.colors = o3d.utility.Vector3dVector(colors)
        self.label = label
        self.world_transformation = transformation # captures all transformations from camera space -> transform_1 -> ... -> transform_n -> world space
        self.timestamp = datetime.now()
    
    @property
    def points(self):
        return np.asarray(self._pcl.points)
    
    @property
    def colors(self):
        return np.asarray(self._pcl.colors)
    
    @points.setter
    def points(self, value):
        self._pcl = o3d.geometry.PointCloud()
        self._pcl.points = o3d.utility.Vector3dVector(value)
        self.timestamp = datetime.now()
    
    @colors.setter
    def colors(self, value):
        self._pcl = o3d.geometry.PointCloud()
        self._pcl.colors = o3d.utility.Vector3dVector(value)
        self.timestamp = datetime.now()

    def is_empty(self):
        return len(self) == 0
    
    def record_transform(self, transformation):
        points_copy = list(self.points)
        colors_copy = list(self.colors)
        new_transformation = transformation @ self.world_transformation
        pcl = PointCloud(points_copy, colors_copy, self.label, new_transformation)
        return pcl
        
    def transform(self, transformation):
        # in place
        self._pcl.transform(transformation)
        return self
    
    def transform_to_rtab(self):
        # in place
        # this permutes the y and z coordinates and flips across the x-axis
        transformation = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
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
        print(self.points.shape)
        print(self.colors.shape)
        temp = np.copy(self.colors)
        
        self.points = self.points[~noise]
        self.colors = temp[~noise]
        
        
        return self
    
    # def save(self, filename=None):
    #     filename = filename or str(self.timestamp).replace(" ", "_") + "_" + self.label
    #     vertex = np.array([(x, y, z) for x, y, z in self.points], 
    #                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #     el = PlyElement.describe(vertex, 'vertex')
    #     PlyData([el]).write(filename)

    def __str__(self):
        return f"{self.label} pointcloud | points: " + str(self.points)
    
    def __add__(self, pcl):
        points = np.vstack((self.points, pcl.points))
        colors = np.vstack((self.colors, pcl.colors))
        return PointCloud(points, colors, self.label, self.world_transformation)
    
    def __len__(self):
        return len(self._pcl.points)