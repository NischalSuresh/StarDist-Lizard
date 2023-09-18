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
    
    def calculate_object_probabilities(self, instance_segmentation_map):
        """
        Calculate object prob for each pixel in the image.
        Args:
        - instance_segmentation_map: A 2D numpy array representing the instance segmentation map.
        Returns:
        - object_probabilities: A 2D numpy array of the same shape as the input image with object probabilities.
        """

        # init the object probs with zeros
        object_probabilities = np.zeros_like(instance_segmentation_map, dtype=float)
        # iterate over each instance
        for inst in np.unique(instance_segmentation_map)[1:]:
            instance_pixels = (instance_segmentation_map == inst)
            boundary_pixels = skimage.segmentation.find_boundaries(instance_pixels, 2, mode='outer', background=0)
            boundary_indices = np.argwhere(boundary_pixels)

            # iterate over each pixel in the instance
            for i in range(instance_segmentation_map.shape[0]):
                for j in range(instance_segmentation_map.shape[1]):
                    if instance_pixels[i,j]:
                        distances_to_background = np.linalg.norm(
                        boundary_indices - np.array([i, j]), axis=1
                        )
                        min_distance = np.min(distances_to_background)
                        # Normalize the distance to [0, 1]
                        object_probabilities[i, j] = min_distance / distances_to_background.max()
        return object_probabilities

    def calculate_star_distances(self, instance_segmentation_map, n_rays=8):
        # Function adapted from StarDist repo https://github.com/stardist/stardist.git
        """
        Calculate object prob for each pixel in the image.
        Args:
        - instance_segmentation_map: A 2D numpy array representing the instance segmentation map.
        - n_rays: Number of rays alobg which the distance is calculated
        Returns:
        - dst: A 3D numpy array with star distances (radial along rays) captured in 3rd dim.
        """    
        n_rays = int(n_rays)
        instance_segmentation_map = instance_segmentation_map.astype(np.uint16,copy=False)
        dst = np.empty(instance_segmentation_map.shape+(n_rays,),np.float32)

        for i in range(instance_segmentation_map.shape[0]):
            for j in range(instance_segmentation_map.shape[1]):
                value = instance_segmentation_map[i,j]
                if value == 0:
                    dst[i,j] = 0
                else:
                    st_rays = np.float32((2*np.pi) / n_rays)
                    for k in range(n_rays):
                        phi = np.float32(k*st_rays)
                        dy = np.cos(phi)
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            ii = int(round(i+x))
                            jj = int(round(j+y))
                            if (ii < 0 or ii >= instance_segmentation_map.shape[0] or
                                jj < 0 or jj >= instance_segmentation_map.shape[1] or
                                value != instance_segmentation_map[ii,jj]):
                                # small correction as we overshoot the boundary
                                t_corr = 1-.5/max(np.abs(dx),np.abs(dy))
                                x -= t_corr*dx
                                y -= t_corr*dy
                                dist = np.sqrt(x**2+y**2)
                                dst[i,j,k] = dist
                                break
        return dst
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        instance_map = self.instance_maps[idx,:,:]
        if self.transforms:
            image = self.transforms(image)
            instance_map = self.transforms(instance_map)
        image = torchvision.transforms.functional.to_tensor(image)
        object_probabilities = self.calculate_object_probabilities(instance_map)
        star_poly_dist = self.calculate_star_distances(instance_map)
        return image, object_probabilities, star_poly_dist
