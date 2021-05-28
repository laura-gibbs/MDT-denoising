import glob
import os
from skimage import io, transform
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import random

class CAEDataset(Dataset):

    def __init__(self, region_dir='../a_mdt_data/HR_model_data/qtrland_training_regions', quilt_dir='./quilting/DCGAN_32deg', transform=None):
            self.region_dir = region_dir
            self.quilt_dir = quilt_dir
            self.transform = transform
            self.paths = glob.glob(os.path.join(region_dir, '*.npy'))
            self.quilt_paths = glob.glob(os.path.join(quilt_dir, '*.png'))
            print(region_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input_name = self.paths[idx]
        random_quilt = random.choice(self.quilt_paths)
        
        # Load and apply random quilt
        target_img = np.load(input_name)
        quilt = Image.open(random_quilt)
        quilt = quilt.convert(mode='L')
        quilt = np.array(quilt).astype(np.float32)
        mask = target_img != 0
        quilt = (quilt - np.nanmin(quilt)) / (np.nanmax(quilt) - np.nanmin(quilt))
        img = target_img + .3* quilt * mask

        # consider turning back to PIL images for transforms e.g. rotations, flips
        # img = Image.fromarray(img)
        # target_img = Image.fromarray(target_img)

        if self.transform is not None:
            img = self.transform(img)
            target_img = self.transform(target_img)

        return ToTensor()(img), ToTensor()(target_img)