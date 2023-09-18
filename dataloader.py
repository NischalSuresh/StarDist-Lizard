import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        instance_map = self.instance_maps[idx,:,:,:]
        if self.transforms:
            image = self.transforms(image)
            instance_map = self.transforms(instance_map)
        
        return image, instance_map
