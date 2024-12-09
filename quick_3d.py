from projections import PointProjector
from PIL import Image
import numpy as np
import cv2
import os
import open3d as o3d
from aggregator import PointCloudAggregator

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

DIRECTORY = "/home/bwilab/Semantic-Mapping-BWI/red_chair"
world_transforms = get_rotation_translation(f"{DIRECTORY}/poses-id.txt")
file_basenames = [os.path.basename(file) for file in os.listdir(f"{DIRECTORY}/rgb")]
file_basenames.sort(key=lambda x: int(x[:-4]))
rgb_images = [f"{DIRECTORY}/rgb/{file}" for file in file_basenames]
depth_images = [f"{DIRECTORY}/depth/{file}" for file in file_basenames]

projector = PointProjector()
aggregator = PointCloudAggregator(eps=0.50, icp_eps=0.05)

pointclouds = []

rgb_images = rgb_images[:15]
depth_images = depth_images[:15]

for i, (rgb_path, depth_path) in enumerate(zip(rgb_images, depth_images)):
    print(f"{i} / {len(rgb_images)}")
    with Image.open(rgb_path) as color_image, Image.open(depth_path) as depth_image:
        ## Make sure images are same dims
        color_image, depth_image = np.array(color_image), np.array(depth_image)
        resized_color_image = cv2.resize(color_image, depth_image.shape[::-1])

    pcl = projector.get_pointcloud(depth_image, color_image)
    transform = world_transforms[os.path.basename(rgb_path)]
    pcl.transform(transform)
    pointclouds.append(pcl)
    if i == 50:
        break

count = 0
while len(pointclouds) > 1:
    print(len(pointclouds))
    new_pointclouds = []
    for i in range(1, len(pointclouds), 2):
        target = pointclouds[i] # 1
        source = pointclouds[i - 1] # 0 
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source._pcl, target._pcl, 0.01, np.eye(4), 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        )
        source.transform(reg_p2p.transformation)
        target += source
        new_pointclouds.append(target)

    pointclouds = new_pointclouds
    count += 1

    projector.visualize(pointclouds)
print(len(pointclouds))
projector.visualize(pointclouds)