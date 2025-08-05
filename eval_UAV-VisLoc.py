import glob
import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as tvf
from tqdm import tqdm
import cv2
from collections import defaultdict
import torch.nn as nn
from models.aggregators.se2gem import se2gem
from models.backbones.e2resnet import E2ResNet



def extract_fields(file_path, img_type):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    fields = base_name.split('@')
    if img_type == 'query':
        num = 0
        longitude = float(fields[1])
        latitude = float(fields[2])
    elif img_type == 'db':
        # num = int(fields[0])
        num = 0
        longitude = float(fields[1])
        latitude = float(fields[2])
    longitude = round(longitude, 10)
    latitude = round(latitude, 10)
    
    return num, longitude, latitude

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, img_path, img_type):
        super().__init__()
        self.img_path = img_path

        # Validate img_type
        if img_type not in ('query', 'db'):
            raise ValueError('img_type should be either "query" or "db"')
        
        # path to images
        img_path_list = glob.glob(self.img_path + '/**/**/*.png', recursive=True)
        
        if img_type == 'query':
            self.img_path_list = img_path_list
        elif img_type == 'db':
            # 过滤只保留编号为0的数据
            self.img_path_list = [x for x in img_path_list if extract_fields(x, 'db')[0] == 0]
            # 排序文件名
            self.img_path_list = sorted(self.img_path_list, key=lambda x: extract_fields(x, 'db'))
            
        assert len(self.img_path_list) > 0, f'No images found in {self.img_path}'

        file_names_list = [os.path.basename(path) for path in self.img_path_list]

        # Extract coordinates and store them in the same order as img_path_list
        if img_type == 'query':
            self.coordinates = [(longitude, latitude) for _, longitude, latitude in (extract_fields(x, 'query') for x in file_names_list)]
        elif img_type == 'db':
            self.coordinates = [(longitude, latitude) for _, longitude, latitude in (extract_fields(x, 'db') for x in file_names_list)]

    def __getitem__(self, index):
        img = load_image(self.img_path_list[index])
        return img, index

    def __len__(self):
        return len(self.img_path_list)

class InferencePipeline:
    def __init__(self, model, dataset, feature_dim, db_type, batch_size=1, num_workers=1, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.db_type = db_type

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'db') -> np.ndarray:

        if os.path.exists(f'./LOGS/global_descriptors_{split}_512_0deg.npy'):
            print(f"Skipping {split} features extraction, loading from cache")
            return np.load(f'./LOGS/global_descriptors_{split}_512_0deg.npy')
            
        self.model.to(self.device)
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                imgs, indices = batch
                imgs = imgs.to(self.device)

                # model inference
                descriptors = self.model(imgs)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors

        # save global descriptors
        if self.db_type == 'db':
            np.save(f'./LOGS/global_descriptors_{split}_512_0deg.npy', global_descriptors)
        return global_descriptors
    
def haversine(coord1, coord2):
    import math
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
def load_image(path):
    image_pil = Image.open(path).convert("RGB")

    # add transforms
    transforms = tvf.Compose([
        tvf.Resize((266, 399), interpolation=tvf.InterpolationMode.BICUBIC),

        tvf.ToTensor(),
        tvf.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    # apply transforms
    image_tensor = transforms(image_pil)
    return image_tensor

class ultravpr(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = E2ResNet(depth=50, out_indices=(3, ), with_geotensor=True, orientation=8, middle_channels=2048)
        self.aggregator = se2gem(in_dim=256, out_dim=256)
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

def load_model(ckpt_path):
    model = ultravpr()
    
    if(ckpt_path != ""):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'], strict=False)
        # model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {ckpt_path} Successfully!")
    model.eval()
   
    return model

def calculate_top_k(q_matrix: np.ndarray,
                    db_matrix: np.ndarray,
                    top_k: int = 10) -> np.ndarray:
    # compute similarity matrix
    similarity_matrix = np.matmul(q_matrix, db_matrix.T)  # shape: (num_query, num_db)

    # compute top-k matches
    top_k_matches = np.argsort(-similarity_matrix, axis=1)[:, :top_k]  # shape: (num_query_images, 10)

    return top_k_matches

def record_matches(top_k_matches: np.ndarray,
                   query_dataset: BaseDataset,
                   database_dataset: BaseDataset,
                   out_file: str = 'record.txt') -> None:
    with open(f'{out_file}', 'a') as f:
        for query_index, db_indices in enumerate(tqdm(top_k_matches, ncols=100, desc='Recording matches')):
            pred_query_path = query_dataset.img_path_list[query_index]
            pred_db_paths_list = [] 
            for i in db_indices.tolist():
                pred_db_paths = database_dataset.img_path_list[i]
                pred_db_paths_list.append(pred_db_paths)
            db_paths_str = ' '.join(pred_db_paths_list)
            f.write(f'{pred_query_path} {db_paths_str}\n')


def recall_rate(top_k_matches: np.ndarray,
                   query_dataset: BaseDataset,
                   database_dataset: BaseDataset) -> None:

    # Initialize counters for recall rates
    recall_counts = defaultdict(int)
    total_queries = len(query_dataset.coordinates)
    
    for query_index, db_indices in enumerate(tqdm(top_k_matches, ncols=100, desc='Recording matches')):
        pred_query_path = query_dataset.img_path_list[query_index]
        pred_query_coordi = query_dataset.coordinates[query_index]
        pred_db_paths_list = []  # 用于存储所有的 pred_db_paths
        pred_db_distance = []
        recall_success = {1: False, 5: False, 10: False}  # Initialize success flags for top1, top5, top10
        
        for rank, i in enumerate(db_indices.tolist(), 1):
            pred_db_paths = database_dataset.img_path_list[i]
            pred_db_coordi = database_dataset.coordinates[i]
            pred_db_paths_list.append(pred_db_paths)
            
            # Calculate distance
            # print('query_coord:    ',pred_query_coordi)
            # print('pred_db_coordi: ',pred_db_coordi)
            distance = haversine(pred_query_coordi, pred_db_coordi)
            pred_db_distance.append(distance)
            
            # Record recall success
            if distance < 200:
                if rank == 1:
                    recall_success[1] = True
                if rank <= 5:
                    recall_success[5] = True
                if rank <= 10:
                    recall_success[10] = True

        
        # Update recall counters
        for k in recall_success:
            if recall_success[k]:
                recall_counts[k] += 1
    
    # Calculate recall rates
    recall_rates = {k: (count / total_queries) * 100 for k, count in recall_counts.items()}
    
    # Print recall rates
    for k in [1, 5, 10]:
        print(f'Top-{k} Recall Rate: {recall_rates[k]:.2f}%')



# load images
     # path to database images folder path
query_path = '/media/robot/CBBPS9/dataset/custom_uav_visloc/custom/01/query'                    # path to query images folder path
datasets_path = '/media/robot/CBBPS9/dataset/custom_uav_visloc/custom/01/reference'


query_dataset = BaseDataset(query_path, 'query')
database_dataset = BaseDataset(datasets_path, 'db')

# load model
model = load_model('./runPath/e2resnet50_c8_se2gem_32/checkpoints/model_best.pth.tar') 

# set up inference pipeline
database_pipeline = InferencePipeline(model=model, dataset=database_dataset, feature_dim=256, db_type = 'db')
query_pipeline = InferencePipeline(model=model, dataset=query_dataset, feature_dim=256, db_type = 'query')

# run inference
db_global_descriptors = database_pipeline.run(split='db')  # shape: (num_db, feature_dim)
query_global_descriptors = query_pipeline.run(split='query')  # shape: (num_query, feature_dim)

# calculate top-k matches
top_k_matches = calculate_top_k(q_matrix=query_global_descriptors, db_matrix=db_global_descriptors, top_k=10)

# record query_database_matches
record_matches(top_k_matches, query_dataset, database_dataset, out_file='./LOGS/record_512.txt')

recall_rate(top_k_matches, query_dataset, database_dataset)