"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import numpy as np
import random
from osgeo import gdalnumeric
import os

def get_dataloader(batch_size, mode='train', shuffle=True):
    # torch.utils.data.DataLoader
    loader = torch.utils.data.DataLoader(
            TianZhi('data/TianZhi', mode=mode),
            batch_size=batch_size,
            num_workers=8,
            shuffle=shuffle)
    return loader

def load_img(path):
    img = gdalnumeric.LoadFile(path)
    #img = np.transpose(img,[1,2,0])
    img = np.array(img, dtype="float")
    img_new = img / 255.0
    return img_new

def load_mask(path):
    mask = gdalnumeric.LoadFile(path)
    mask = np.array(mask).astype(np.float)
    h, w = mask.shape[1:]
    mask[mask > 0] = 1
    return np.max(mask, axis=0).reshape(1, h, w)

def merge_change_mask(A, B):
    h, w = A.shape[1:]
    assert(B.shape[1] == h and B.shape[2] == w)
    mask = np.zeros((2, h, w), dtype=np.float)
    single_A = np.max(A, axis=0)
    single_B = np.max(B, axis=0)
    
    mask[0, :, :][single_A > 0] = 1
    mask[1, :, :][single_B > 0] = 1
    return mask

class TianZhi(data.Dataset):
    def __init__(self, root, mode='train'):
        self.data_root = root
        self.mode = mode
        self.imageid_list = []
        img_dir_A = os.path.join(root, mode, 'Image1')
        img_dir_B = os.path.join(root, mode, 'Image2')
        for fname in os.listdir(img_dir_A):
            if(fname.endswith('png')):
                assert(os.path.exists(os.path.join(img_dir_B, fname)))
                self.imageid_list.append(fname.split('.')[0])

    def read_data(self, image_id):
        image_A = load_img(os.path.join(self.data_root, self.mode, 'Image1', image_id+'.png'))
        image_B = load_img(os.path.join(self.data_root, self.mode, 'Image2', image_id+'.png'))
        change_mask_A = load_mask(os.path.join(self.data_root, self.mode, 'label1', image_id+'.png'))
        change_mask_B = load_mask(os.path.join(self.data_root, self.mode, 'label2', image_id+'.png'))
        name_A = os.path.join('label1', image_id+'.png')
        name_B = os.path.join('label2', image_id+'.png')
        
        if(self.mode == 'train'):
            building_mask_A = load_mask(os.path.join(self.data_root, self.mode, 'new_gt1', image_id+'.png'))
            building_mask_B = load_mask(os.path.join(self.data_root, self.mode, 'new_gt2', image_id+'.png'))
            # condider add augmentation here
            # Random crop
            #std_size = 1024
            #h, w = image_A.shape[1:]
            #offset_y = int(random.random() * (h - std_size))
            #offset_x = int(random.random() * (w - std_size))
            #image_A = image_A[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]
            #image_B = image_B[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]
            #change_mask_A = change_mask_A[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]
            #change_mask_B = change_mask_B[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]
            #building_mask_A = building_mask_A[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]
            #building_mask_B = building_mask_B[:, offset_y:offset_y+std_size, offset_x:offset_x+std_size]

            if(np.max(change_mask_A) > 0 and np.max(change_mask_B) > 0):
                if(random.random() < 0.5):
                    return [image_A, image_B, building_mask_A, building_mask_B, change_mask_A]
                else:
                    return [image_B, image_A, building_mask_B, building_mask_A, change_mask_B]
            elif(np.max(change_mask_A) > 0):
                return [image_A, image_B, building_mask_A, building_mask_B, change_mask_A]
            elif(np.max(change_mask_B) > 0):
                return [image_B, image_A, building_mask_B, building_mask_A, change_mask_B]
            else:
                if(random.random() < 0.5):
                    return [image_A, image_B, building_mask_A, building_mask_B, change_mask_A]
                else:
                    return [image_B, image_A, building_mask_B, building_mask_A, change_mask_B]

        #mask=np.expand_dims(mask,axis=0)
        else:
            return [image_A, image_B, change_mask_A, change_mask_B]

    def __getitem__(self, index):
        image_id = self.imageid_list[index]
        elems = self.read_data(image_id)

        for i in range(len(elems)):
            elems[i] = torch.Tensor(elems[i].copy())
        return elems

    def __len__(self):
        return len(self.imageid_list)
