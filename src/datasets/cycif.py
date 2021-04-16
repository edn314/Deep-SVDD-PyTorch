import os
import numpy as np
import torch
from glob import glob
from torch.utils import data
from PIL import Image
from imageio import imread
import cv2
from base.torchvision_dataset import TorchvisionDataset

class CyCIF_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        self.train_set = MyCyCIF(root=root,split="train")
        self.val_set = MyCyCIF(root=root,split="val")
        self.test_set = MyCyCIF(root=root,split="test")

class MyCyCIF(data.Dataset):
    def __init__(self,root,split="train"):
        super().__init__()
        self.root = root
        self.split = split
        self.files = {}

        fpattern = os.path.join(self.root, f'{self.split}_split/*/*.png')
        self.files = sorted(glob(fpattern))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        file_paths = self.files
        
        # Val mode
        if self.split == "val" or self.split == "test":
            # Image
            fpath = file_paths[index]

            image = np.asarray(imread(fpath))
            image = self.normalize_images(image) #,train=False)

            torch_image = torch.from_numpy(image)
            torch_image = torch_image.type(torch.FloatTensor)

            if (os.path.basename(os.path.dirname(file_paths[index])) == 'clean'):
                binary_label = 0
                # Normal Label for segmentation
                label = np.zeros((256,256), dtype = np.uint8)
                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0)
            else:
                # Anomaly label for detection
                binary_label = 1
                # Anomaly Labels for segmentation
                ann_path = '/n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/'
                full_file = os.path.split(file_paths[index])[1]
                file_name = full_file.split("_")[0]
                channel_num = int(full_file.split("_")[1].split("c")[1])

                label_path = os.path.join(ann_path,f'{file_name}-c{channel_num:04d}.png')
                
                # Scale by upsampling factor = 16 to get coordinates
                label = np.asarray(imread(label_path))
                loc_y = int(full_file.split("_")[2].split("y")[1]) // 16
                loc_x = int(full_file.split("_")[3].split("x")[1]) // 16

                # Original label size is 1024, extract patch, scale to 256
                label_crop = label[loc_y: loc_y + 1024//16, loc_x: loc_x + 1024//16]
                
                label = cv2.resize(label_crop,(label_crop.shape[0]*4,label_crop.shape[1]*4)) # upsample to 256 x 256
                
                # import pdb; pdb.set_trace()
                # import torchvision
                # torchvision.utils.save_image(torch.unsqueeze(torch.from_numpy(label),dim=0),
                #                 "./label_test2.png",
                #                 normalize=True,
                #                 nrow = 1)
                # torchvision.utils.save_image((torch.from_numpy(image)),
                #                 "./img_test2.png",
                #                 normalize=True,
                #                 nrow = 1)

                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0) 
   
            return torch_image, binary_label, index
        
        # Train mode
        else:
            
            fpath = file_paths[index]
            image = np.asarray(imread(fpath))

            image = self.normalize_images(image) #,train=True)

            torch_image = torch.from_numpy(image)

            torch_image = torch_image.type(torch.FloatTensor)
            binary_label = 0 # training set contains only clean images

            return torch_image, binary_label, index

    def normalize_images(self,images): #,train):
            # Resize Operation
            images = self.resize(images)
            if len(np.shape(images)) != 3:
                images = np.expand_dims(images,axis=2)
            images = np.asarray(images)

            images = images / 255 # between [0,1]
            images = np.transpose(images, [2, 0, 1]) # Images with [C x H x W]

            return images

    def resize(self, image, shape=(256, 256)):
        return cv2.resize(image,shape)