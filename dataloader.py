
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
import glob


class TEETHdataset(Dataset):
    def __init__(self, transform_type = "resize", mode ="train", test_dir =None):

        self.mode = mode
        if self.mode == "test":
            self.test_img_path = test_dir
            self.test_list = glob.glob(os.path.join(self.test_img_path,"*.jpg"))

        
        json_path = '/home/ubuntu/researches/pasamedi/data/images/kjh-teeth-labeling/manifests/output/output.manifest'

        if not os.path.exists(json_path):
            raise Exception("No json_path")

        with open(json_path) as f:
            file = f.readlines()
            pair_list = []
            for line in file:
                pair_list.append(json.loads(line))        
        
        self.pair_list = pair_list
        self.img_dir_path = '/home/ubuntu/researches/pasamedi/data/images'
        self.mask_dir_path = '/home/ubuntu/researches/pasamedi/data/images/kjh-teeth-labeling/annotations/consolidated-annotation/output'



        if not os.path.exists(self.img_dir_path):
            raise Exception("No img_path")
        if not os.path.exists(self.mask_dir_path):
            raise Exception("No mask_path")

        self.transform_type = transform_type

        if transform_type=="resize":
            self.img_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
            self.mask_transform = transforms.Compose([transforms.Resize((224,224), interpolation=Image.NEAREST)])

        self.totensor = transforms.ToTensor()

    def __getitem__(self , idx):

        if self.mode == "test":
            img_path = self.test_list[idx]
            img = Image.open(img_path)
            mask = torch.tensor([0])
            if self.transform_type=="resize": # convert image size->(224,224)
                img_array = np.array(img)
                img = self.img_transform(img)
                
            elif self.transform_type=="original":# use original size of image
                img = self.totensor(img)

            elif self.transform_type=="crop":
                pass            
        elif self.mode=="train":
            pair = self.pair_list[idx]
            img_name = os.path.basename(pair['source-ref'])
            mask_name = os.path.basename(pair['kjh-teeth-labeling-ref'])
            if self.mode == 'train':
                img_path = os.path.join(self.img_dir_path, img_name)
            elif self.mode == 'test':
                img_path = self.test_list[idx]
            mask_path = os.path.join(self.mask_dir_path, mask_name)

            img = Image.open(img_path)
            mask = Image.open(mask_path)

            if self.transform_type=="resize": # convert image size->(224,224)
                img_array = np.array(img)
                img = self.img_transform(img)
                

                mask = np.array(mask)
                
                mask = torch.as_tensor(mask, dtype=torch.uint8)
                mask = mask.unsqueeze(0)
                mask = self.mask_transform(mask)
                mask = mask.squeeze(0)
                mask = mask.type(dtype=torch.LongTensor)
            elif self.transform_type=="original":# use original size of image
                img = self.totensor(img)
                mask = np.array(mask)
                mask = torch.from_numpy(mask)
                mask = mask.type(dtype=torch.LongTensor)
            elif self.transform_type=="crop":
                pass

            

        return img, mask


    def __len__(self):
        if self.mode == "train":
            return len(self.pair_list)
        elif self.mode == "test":
            return len(self.test_list)
def test():
    #Dataloader#
    dataset = TEETHdataset(resize = False)
    dataloader = DataLoader(dataset, batch_size = 20, shuffle = True)

    img, mask = next(iter(dataloader))
    


if __name__=="__main__":
    test()



