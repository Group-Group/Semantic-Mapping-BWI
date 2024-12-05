from projections import PointProjector
from PIL import Image
import numpy as np
import cv2
import os
import open3d as o3d

# test = o3d.geometry.PointCloud()
# test.points = o3d.utility.Vector3dVector(np.array([[1, 2, 3]]))


def get_rotation_translation(filepath) -> dict[np.ndarray]: ## filename -> transform
    world_transforms = dict()
    
    skip_header = True
    with open(filepath, 'r') as file:
        for line in file:
            if skip_header:
                skip_header = False
                continue

            data = line.split()[1:] # ignore timestamp
            filename = data.pop() + ".png"
            data = [float(p) for p in data]
            rigid_transform = quaternion_to_rigid_transform(*data)
            
            ## first do E, then try E inverse
            E = np.eye(4)
            E[:3] = rigid_transform
            world_transforms[filename] = E

    return world_transforms

def quaternion_to_rigid_transform(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    # Normalize the quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Compute the rotation matrix
    E_inv = np.zeros((3, 4))
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])

    E_inv[:3, :3] = R
    E_inv[:, 3] = np.array([x, y, z])
    return E_inv

DIRECTORY = '/home/justin/FRI/Semantic-Mapping-BWI/red_chair/'
world_transforms = get_rotation_translation(f"{DIRECTORY}/poses-id.txt")
color_images = [f"{DIRECTORY}/rgb/{file}" for file in os.listdir(f"{DIRECTORY}/rgb")] ## os.listdir is arbitrary
color_images.sort(key=lambda x: int(os.path.basename(x)[:-4]))

depth_images = [f"{DIRECTORY}/depth/{file}" for file in os.listdir(f"{DIRECTORY}/depth")]
depth_images.sort(key=lambda x: int(os.path.basename(x)[:-4]))

projector = PointProjector()
pointclouds = []
for i, (rgb_path, depth_path) in enumerate(zip(color_images, depth_images)):
    print(f"{i} / {len(color_images)}")
    with Image.open(rgb_path) as color_image, Image.open(depth_path) as depth_image:
        ## Make sure images are same dims
        color_image, depth_image = np.array(color_image), np.array(depth_image)
        resized_color_image = cv2.resize(color_image, depth_image.shape[::-1])
    transform = world_transforms[os.path.basename(rgb_path)]

    pcl = projector.get_pointcloud(depth_image, color_image)
    # pcl.transform(transform)
    pointclouds.append(pcl)
    break

projector.visualize(pointclouds)
