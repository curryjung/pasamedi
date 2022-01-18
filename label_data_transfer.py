import glob
import os
import json
from PIL import Image
from shutil import copyfile

save_dir = "./data/images/lable_imgs"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

json_path = '/home/ubuntu/researches/pasamedi/data/images/kjh-teeth-labeling/manifests/output/output.manifest'

if not os.path.exists(json_path):
    raise Exception("No json_path")

with open(json_path) as f:
    file = f.readlines()
    pair_list = []
    for line in file:
        pair_list.append(json.loads(line))  

for pair in pair_list:
    img_name = os.path.basename(pair['source-ref'])
    img_name = os.path.splitext(img_name)[0]
    ref_name = os.path.basename(pair['kjh-teeth-labeling-ref'])

    ref_path = os.path.join('/home/ubuntu/researches/pasamedi/data/images/kjh-teeth-labeling/annotations/consolidated-annotation/output',ref_name)

    copyfile(ref_path, os.path.join(save_dir,img_name+'.png'))