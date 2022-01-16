
from configparser import Interpolation
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


class TEETHdataset(Dataset):
    def __init__(self):
        
        json_path = './data/kjh-teeth-labeling/manifests/intermediate/1/output.manifest'

        if not os.path.exists(json_path):
            raise Exception("No json_path")

        with open(json_path) as f:
            file = f.readlines()
            pair_list = []
            for line in file:
                pair_list.append(json.loads(line))        
        
        self.pair_list = pair_list
        self.img_dir_path = './data'
        self.mask_dir_path = './data/kjh-teeth-labeling/annotations/consolidated-annotation/output'

        if not os.path.exists(self.img_dir_path):
            raise Exception("No img_path")
        if not os.path.exists(self.mask_dir_path):
            raise Exception("No mask_path")

        self.img_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
        self.mask_transform = transforms.Compose([transforms.Resize((224,224), interpolation=Image.NEAREST)])

    def __getitem__(self , idx):

        pair = self.pair_list[idx]
        img_name = os.path.basename(pair['source-ref'])
        mask_name = os.path.basename(pair['kjh-teeth-labeling-ref'])
        img_path = os.path.join(self.img_dir_path, img_name)
        mask_path = os.path.join(self.mask_dir_path, mask_name)

        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_array = np.array(img)
        img = self.img_transform(img)
        

        mask = np.array(mask)
        
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.unsqueeze(0)
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0)
        mask = mask.type(dtype=torch.LongTensor)


        return img, mask


    def __len__(self):
        return len(self.pair_list)

def test():
    #Dataloader#
    dataset = TEETHdataset()
    dataloader = DataLoader(dataset, batch_size = 20, shuffle = True)

    img, mask = next(iter(dataloader))

    print(img.shape)
    print(mask.shape)



if __name__=="__main__":
    test()



