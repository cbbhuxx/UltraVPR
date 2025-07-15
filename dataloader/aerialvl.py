import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import re
import torch.nn as nn

import torch
from glob import glob
from os.path import join
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

device = torch.device("cuda")

DIM =256

train_q_dir = "/media/robot/UD210/Datasets/images/queries/"
train_db_dir = '/media/robot/UD210/Datasets/images/database/'
train_n_dir = '/media/robot/UD210/Datasets/images/database/'

test_db_dir = '/media/robot/CBBPS9/dataset/VPAir/reference_views/'
test_q_dir = "/media/robot/CBBPS9/dataset/VPAir/queries/"

def input_transform():

    return transforms.Compose([
        transforms.RandomRotation([-180,180], expand=False),
        transforms.RandomResizedCrop(size=400, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC)
                              ])

def input_transform_test():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        transforms.Resize((300, 400), interpolation=transforms.InterpolationMode.BICUBIC)
                              ])

def input_transform_neg():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC)
                              ])

def get_training_classes_set():
    return ClassesDatasetFromStruct(root_q_dir=train_q_dir, root_db_dir=train_db_dir, input_transform=input_transform_neg())

def get_training_negativies_set(root_dir=None):
    return negativiesDatasetFromStruct(root_dir=root_dir, input_transform=input_transform_neg())

def get_training_query_set(vpr_model=None):
    return QueryDatasetFromStruct(root_db_dir=train_db_dir,root_q_dir=train_q_dir,root_n_dir=train_n_dir,
                                  input_transform=input_transform(), model=vpr_model)
def get_whole_val_set():
    return ValDatasetFromStruct(root_dir=test_db_dir,
                                input_transform=input_transform_test())

def get_whole_test_set():
    return ValDatasetFromStruct_(root_dir=test_q_dir,
                                  input_transform=input_transform_test())

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = data.dataloader.default_collate(negatives)
    return query, positive, negatives, negCounts, indices

class ClassesDatasetFromStruct(data.Dataset):
    def __init__(self, root_q_dir, root_db_dir, input_transform=None):
        self.root_q_dir = root_q_dir
        self.root_db_dir = root_db_dir
        self.input_transform = input_transform
        self.dataset = "aerialvl"
        self.whichSet = "negativies"
        self.query_info = sorted(glob(join(self.root_q_dir, '*.png'), recursive = True))
        self.ref_info = sorted(glob(join(self.root_db_dir, '*.png'), recursive = True))
        self.images_info = self.query_info + self.ref_info
    def __getitem__(self, index):
        image = Image.open(self.images_info[index])
        image = image.convert('RGB')
        if self.input_transform:
            image = self.input_transform(image) 
        return image, index
    def __len__(self):
        return len(self.images_info)

class negativiesDatasetFromStruct(data.Dataset):
    def __init__(self, root_dir, input_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.dataset = "aerialvl"
        self.whichSet = "negativies"
        self.images_info = sorted(glob(join(self.root_dir, '*.png'), recursive = True))    
    def __getitem__(self, index):
        image = Image.open(self.images_info[index])
        image = image.convert('RGB')
        if self.input_transform:
            image = self.input_transform(image) 
        return image, index
    def __len__(self):
        return len(self.images_info)


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, root_db_dir, root_q_dir, root_n_dir, input_transform=None, model=None):
        self.root_db_dir = root_db_dir
        self.root_q_dir = root_q_dir
        self.root_n_dir = root_n_dir
        self.input_transform = input_transform
        self.dataset = "aerialvl"
        self.whichSet = "train"
        self.model = model
        self.positive_info = []
        self.database_paths = sorted(glob(join(root_db_dir, '*.png'), recursive = True))
        self.query_info = sorted(glob(join(root_q_dir, '*.png'), recursive = True))
        self.negatives_info = sorted(glob(join(root_db_dir, '*.png'), recursive = True))
        
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.query_info]).astype(float)

        knn = NearestNeighbors(n_neighbors=30, n_jobs = -1)
        knn.fit(self.database_utms)
        _, self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             return_distance = True, sort_results=True)
        self._get_positive_info()
        self.isnear = False
        if(self.model!=None):
            self.query_set = get_training_negativies_set(self.root_q_dir)
            self.database_set = get_training_negativies_set(self.root_db_dir)
            query_data_loader = DataLoader(dataset=self.query_set, num_workers=13, 
                                           batch_size=24, shuffle=False, pin_memory=False)
            database_data_loader = DataLoader(dataset=self.database_set, num_workers=13, 
                                           batch_size=24, shuffle=False, pin_memory=False)
            self.model.eval()
            with torch.no_grad():
                print("====> Negative sample mining ...")
                qFeat = np.empty((len(self.query_set), DIM))
                dbFeat = np.empty((len(self.database_set), DIM))
                for iteration, (input, indices) in enumerate(tqdm(database_data_loader, desc="database_data:", unit="item", ncols=80), 1):
                    input = input.to(device)
                    image_encoding = model.backbone(input)
                    aggregate_encoding = model.aggregator(image_encoding)
                    dbFeat[indices.detach().numpy(), :] = aggregate_encoding.detach().cpu().numpy()
                    del input, aggregate_encoding
                for iteration1, (input, indices) in enumerate(tqdm(query_data_loader, desc="query_data:", unit="item", ncols=80), 1):
                    input = input.to(device)
                    image_encoding = model.backbone(input)
                    aggregate_encoding = model.aggregator(image_encoding)
                    qFeat[indices.detach().numpy(), :] = aggregate_encoding.detach().cpu().numpy()
                    del input, aggregate_encoding
            del query_data_loader, database_data_loader
            qFeat = qFeat.astype('float32')
            dbFeat = dbFeat.astype('float32')
            faiss_index = faiss.IndexFlatL2(DIM)
            faiss_index.add(dbFeat)
            _, self.predictions = faiss_index.search(qFeat, 500)
            
    def __getitem__(self, index):
        query = Image.open(self.query_info[index])
        positive = Image.open(self.positive_info[index])
        
        query = query.convert('RGB')
        positive = positive.convert('RGB')
        
        negatives_imgs = []
        if self.model!=None:
            for p in range(500):
                if self.predictions[index][p] not in self.soft_positives_per_query[index][:30]:
                    negatives = Image.open(self.database_paths[self.predictions[index][p]])
                    negatives = negatives.convert('RGB')
                    negatives_imgs.append(negatives)
                    if len(negatives_imgs) == 10:
                        break
        else:
            for k in range(10):
                negatives = Image.open(self.database_paths[self.soft_positives_per_query[index][k+20]])
                negatives = negatives.convert('RGB')
                negatives_imgs.append(negatives)

        transformed_negatives = []
        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)
            for img in negatives_imgs:
                transformed_img = self.input_transform(img)
                transformed_negatives.append(transformed_img)
            negatives_imgs = transformed_negatives

        return query, positive, torch.stack(negatives_imgs), index

    def __len__(self):
        return len(self.query_info)

    def _get_positive_info(self):
        for i in range(len(self.query_info)):
            self.positive_info.append(self.database_paths[self.soft_positives_per_query[i][0]])


class ValDatasetFromStruct(data.Dataset):
    def __init__(self, root_dir, input_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.img_info = []
        self.cc_index = []
        self._get_img_info()
        self.dataset = "aerialvl"
        self.num = len(self.img_info)

    def __getitem__(self, index):
        img = Image.open(self.img_info[index])
        img = img.convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png"):
                    path_img = os.path.join(root, file)
                    num_th = re.findall("\d+", file)[-1]
                    num_th = int(num_th.lstrip('0'))
                    self.cc_index.append(num_th)
                    self.img_info.append(path_img)


class ValDatasetFromStruct_(data.Dataset):
    def __init__(self, root_dir, input_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.img_info = []
        self.cc_index = []
        self._get_img_info()
        self.dataset = "aerialvl"
        self.num = len(self.img_info)

    def __getitem__(self, index):
        img = Image.open(self.img_info[index])
        img = img.convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png"):
                    path_img = os.path.join(root, file)
                    num_th = re.findall("\d+", file)[-1]
                    num_th = int(num_th.lstrip('0'))
                    self.cc_index.append(num_th)
                    self.img_info.append(path_img)


# a = get_training_query_set()

# a.__getitem__(1000)