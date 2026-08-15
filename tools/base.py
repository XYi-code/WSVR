from PIL import Image, ImageFile
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import random
import torchvision.transforms as T
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids = []

        for _, pid, _, _, _ in data:
            pids += [pid]

        pids = set(pids)

        num_pids = len(pids)

        num_imgs = len(data)

        return num_pids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs= self.get_imagedata_info(train)
        num_query_pids, num_query_imgs= self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs= self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, occlusion_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.occlusion_transform = occlusion_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid= self.dataset[index]
        
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        print(f"Transformed image size: {img.size}")

        if self.occlusion_transform:
            original_img, mask, occluded_img = self.occlusion_transform(img)
            print(f"Original image size after occlusion: {original_img.size}")
           
        else:
            original_img = img
            mask = torch.zeros_like(img)
            occluded_img = img 
        return original_img, occluded_img, mask, pid, img_path.split('/')[-1]