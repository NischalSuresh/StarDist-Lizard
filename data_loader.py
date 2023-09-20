import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import euclidean
import skimage

class MyDataset(Dataset):
    ''' - Takes the path of the image file and labels as constructor arguments \n
    - getitem returns image, obj_prob and star_poly_dist
    - For definition of obj_prob and star_poly_dist refer paper'''

    def __init__(self, images_path, labels_path, transforms = None):
        self.images = np.load(images_path)
        self.instance_maps = np.load(labels_path)[:,:,:,0]
        self.transforms = transforms
    
    def __len__(self):
        return self.images.shape[0]
    
    def calculate_starGT(self, instance_segmentation_map, n_rays = 8):
        """
        Calculate object prob for each pixel in the image.
        Args:
        - instance_segmentation_map: A 2D numpy array representing the instance segmentation map.
        - n_rays: The number of rays to sample for each instance.
        Returns:
        - object_probabilities: A 2D numpy array of the same shape as the input image with object probabilities.
        - star_distances: A 3D numpy array of shape (H, W, n_rays) with distances to the n_rays star points.
        - angles: A 3D numpy array of shape (H, W, n_rays) with angles to the n_rays star points.
        """

        # init the object probs with zeros
        object_probabilities = np.zeros_like(instance_segmentation_map, dtype=float)
        star_distances = np.zeros((instance_segmentation_map.shape[0],instance_segmentation_map.shape[1],n_rays), dtype=float)
        angles = np.zeros_like(star_distances, dtype=float)
        # iterate over each instance
        for inst in np.unique(instance_segmentation_map)[1:]:
            instance_pixels = (instance_segmentation_map == inst)
            boundary_pixels = skimage.segmentation.find_boundaries(instance_pixels, 2, mode='inner', background=0)
            boundary_indices = np.argwhere(boundary_pixels)
            # randomly sample n_rays points on the boundary
            n_boundary_pixels = boundary_indices.shape[0]
            try:
                ray_indices = np.random.choice(n_boundary_pixels, size=n_rays, replace=False)
            # if there are less boundary pixels than rays, sample with replacement
            except:
                ray_indices = np.random.choice(n_boundary_pixels, size=n_rays, replace=True)

            ray_points = boundary_indices[ray_indices]
            # iterate over each pixel in the instance
            for i in range(instance_segmentation_map.shape[0]):
                for j in range(instance_segmentation_map.shape[1]):
                    if instance_pixels[i,j]:
                        distances_to_background = np.linalg.norm(
                        boundary_indices - np.array([i, j]), axis=1
                        )
                        min_distance = np.min(distances_to_background)
                        # normalize to [0, 1]
                        object_probabilities[i, j] = min_distance / distances_to_background.max()
                        # calculate the distance from instace pixel to each ray point using np.linalg.norm
                        star_distances[i,j] = np.linalg.norm(ray_points - np.array([i, j]), axis=1)
                        # calc the angles with respect to the x-axis
                        angles_ray = np.arctan2(ray_points[:, 1] - j, ray_points[:, 0] - i)
                        # angles from [-pi,pi] --> [0, 2pi] and normalize to [0, 1]
                        angles[i,j] = ((angles_ray + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)               
        return object_probabilities, star_distances, angles 
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        instance_map = self.instance_maps[idx,:,:]
        if self.transforms:
            image = self.transforms(image)
            instance_map = self.transforms(instance_map)
        image = torchvision.transforms.functional.to_tensor(image)
        object_probabilities, star_distances, angles = self.calculate_starGT(instance_map)
        return image, object_probabilities, star_distances, angles
