"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform
        # ===========================================================================================
        # Computing length once
        # ===========================================================================================
        self.real_len = sum([1 for name in self.real_image_names if os.path.isfile(os.path.join(self.root_path, 'real', name))])
        self.fake_len = sum([1 for name in self.fake_image_names if os.path.isfile(os.path.join(self.root_path, 'fake', name))])
        # directories = ['fake', 'real']
        # tot_size = 0
        # for directory in directories:
        #     tot_size += sum([1 for name in os.listdir(os.path.join(self.root_path, directory)) if
        #                      os.path.isfile(os.path.join(self.root_path, directory, name))])
        # self.length = tot_size

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        if index < self.real_len:
            image = Image.open(os.path.join(self.root_path, 'real', self.real_image_names[index]))
            label = 0
        else:
            image = Image.open(os.path.join(self.root_path, 'fake', self.fake_image_names[index-self.real_len]))
            label = 1
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.real_len + self.fake_len
