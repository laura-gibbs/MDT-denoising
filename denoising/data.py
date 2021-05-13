import glob
import os
from skimage import io, transform
from PIL import Image
from torchvision.transforms import ToTensor


class CAEDataset(Dataset):

    def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.input_paths = glob.glob(os.path.join(root_dir, 'inputs/*.png'))
            self.target_paths = glob.glob(os.path.join(root_dir, 'targets/*.png'))
            print(root_dir)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_name = self.input_paths[idx]
        target_name = self.input_paths[idx]

        input_img = Image.open(input_name)
        target_img = Image.open(target_name)

        if self.transform is not None:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return ToTensor()(input_img), ToTensor()(target_img)