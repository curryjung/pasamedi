import json
import os

json_path = '/home/ubuntu/researches/pasamedi/data/teeth_imgs/teeth-labeling-job/manifests/output/output.manifest'

if not os.path.exists(json_path):
    raise Exception("No json_path")

# with open(json_path) as f:
#     file = f.readlines()
#     pair_list = []
#     for line in file:
#         pair_list.append(json.loads(line))   

with open(json_path,'r',encoding='UTF-8') as f:
    file = f.readlines()



print(5)   