import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import euclidean
import cv2 as cv

class MyDataset(Dataset):
    ''' - Takes the path of the image file and labels as constructor arguments \n
    - getitem returns image, obj_prob and star_poly_dist
    - For definition of obj_prob and star_poly_dist refer paper'''

    def __init__(self, images_path, labels_path, n_rays = 32, transforms = None):
        self.images = np.load(images_path)
        self.instance_maps = np.load(labels_path)[:,:,:,0]
        self.transforms = transforms
        self.n_rays = n_rays
    
    def __len__(self):
        return self.images.shape[0]
    
    def order_contour(self, contour_init):
        """
            takes in the contour and orders it starting from the bottom pixel
            Args:
            - contout_init: A 2D numpy array representing the contour of the instance.
            Returns:
            - rotated_arr: A 2D numpy array of the same shape as the input contour with the bottom pixel as the first element.
            """
        contour_init = contour_init[:, [1, 0]] # cv 
        max_first_index = np.max(contour_init[:, 0])
        # Create a boolean mask to identify rows with the maximum first index
        filtered_data = contour_init[:, 0] == max_first_index
        min_second_index = np.min(contour_init[filtered_data, 1])
        # find index of max first index and min second index
        index = np.where((contour_init[:, 0] == max_first_index) & (contour_init[:, 1] == min_second_index))[0][0]
        rotated_arr = np.concatenate((contour_init[index:], contour_init[:index]))
        return rotated_arr
    
    def calculate_starGT(self,instance_segmentation_map, n_rays):
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
            object_probabilities = np.zeros_like(instance_segmentation_map, dtype=np.float32)
            star_distances = np.zeros((instance_segmentation_map.shape[0],instance_segmentation_map.shape[1],n_rays), dtype=np.float32)
            angles = np.zeros_like(star_distances, dtype=np.float32)
            # iterate over each instance
            for inst in np.unique(instance_segmentation_map)[1:]:
                instance_pixels = (instance_segmentation_map == inst)
                contour_initial, _ = cv.findContours(instance_pixels.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contour_initial = contour_initial[0][:, 0, :]
                contour = self.order_contour(contour_initial)
                # randomly sample n_rays points on the boundary
                contour_n = contour[np.linspace(0, len(contour) - 1, n_rays).astype(int)]
                # iterate over each pixel in the instance
                for i in range(instance_segmentation_map.shape[0]):
                    for j in range(instance_segmentation_map.shape[1]):
                        if instance_pixels[i,j]:
                            distances_to_background = np.linalg.norm(
                            contour - np.array([i, j]), axis=1
                            )
                            min_distance = np.min(distances_to_background)
                            # normalize to [0, 1]
                            if distances_to_background.max() == 0:
                                # handle division by 0
                                object_probabilities[i, j] = 0.0  # Or any other appropriate value
                            else:
                                object_probabilities[i, j] = min_distance /  distances_to_background.max()
                            # calculate the distance from instace pixel to each ray point using np.linalg.norm
                            star_distances[i,j] = np.linalg.norm(contour_n - np.array([i, j]), axis=1)
                            # calc the angles with respect to the j-axis 
                            angles_ray = np.arctan2(contour_n[:, 1] - j, contour_n[:, 0] - i)
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
        object_probabilities, star_distances, angles = self.calculate_starGT(instance_map, self.n_rays)
        return image, object_probabilities, star_distances, angles
