from projections import PointProjector
from PIL import Image
import numpy as np
import cv2
import os
import open3d as o3d
from aggregator import PointCloudAggregator


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

DIRECTORY = "/home/bwilab/Semantic-Mapping-BWI/red_chair"
world_transforms = get_rotation_translation(f"{DIRECTORY}/poses-id.txt")
color_images = [f"{DIRECTORY}/rgb/{file}" for file in os.listdir(f"{DIRECTORY}/rgb")] ## os.listdir is arbitrary
color_images.sort(key=lambda x: int(os.path.basename(x)[:-4]))

depth_images = [f"{DIRECTORY}/depth/{file}" for file in os.listdir(f"{DIRECTORY}/depth")]
depth_images.sort(key=lambda x: int(os.path.basename(x)[:-4]))

projector = PointProjector()
aggregator = PointCloudAggregator(eps=2.0) ## higher eps == more merging, lower eps == more detail (or noise)
pointclouds = []
for i, (rgb_path, depth_path) in enumerate(zip(color_images, depth_images)):
    print(f"{i} / {len(color_images)}")
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
    eps = 0.0001 * (10 ** count)
    for i in range(1, len(pointclouds), 2):
        target = pointclouds[i] # 1
        source = pointclouds[i - 1] # 0 

        # transform = world_transforms[os.path.basename(color_images[i])]
        # pointclouds[i].transform(transform)
        
        source_transform = world_transforms[os.path.basename(color_images[i - 1])] if count == 0 else np.eye(4)
        # print('source_transform' + os.path.basename(color_images[i - 1]))
        
        target_transform = world_transforms[os.path.basename(color_images[i])] if count == 0 else np.eye(4)
        # print("target transform" + os.path.basename(color_images[i]))
        eps += i / 1000
        
        # target.transform(target_transform)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source._pcl, target._pcl, eps, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        source.transform(reg_p2p.transformation)
        target += source
        new_pointclouds.append(target)

    pointclouds = new_pointclouds
    count += 1

    projector.visualize(pointclouds)
print(len(pointclouds))
projector.visualize(pointclouds)
# projector.visualize(aggregator.main)
