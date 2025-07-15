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
import utils
from main import parser
import torch.nn as nn
from models.aggregators.se2gem import se2gem
from models.backbones.e2resnet import E2ResNet

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

        # path to images
        if 'queries' in self.img_path:
            img_path_list = glob.glob(self.img_path + '/*.png', recursive=True)
            self.img_path_list = img_path_list
        elif 'reference_views' in self.img_path:
            img_path_list = glob.glob(self.img_path + '/*.png', recursive=True)
            # sort images for db
            self.img_path_list = sorted(img_path_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            raise ValueError('img_path should be either query or db')
        assert len(self.img_path_list) > 0, f'No images found in {self.img_path}'
    def __getitem__(self, index):
        img = load_image(self.img_path_list[index])
        return img, index
    def __len__(self):
        return len(self.img_path_list)


class InferencePipeline:
    def __init__(self, model, dataset, feature_dim, batch_size=1, num_workers=1, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'db') -> np.ndarray:

        if os.path.exists(f'./LOGS/global_descriptors_{split}.npy'):
            print(f"Skipping {split} features extraction, loading from cache")
            return np.load(f'./LOGS/global_descriptors_{split}.npy')

        self.model.to(self.device)
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                imgs, indices = batch
                imgs = imgs.to(self.device)
                descriptors = self.model(imgs)
                descriptors = descriptors.detach().cpu().numpy()
                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors

        # save global descriptors
        np.save(f'./LOGS/global_descriptors_{split}.npy', global_descriptors)
        return global_descriptors


def load_image(path):

    image_pil = Image.open(path).convert("RGB")

    # add transforms
    transforms = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225]),
        tvf.Resize((300, 400), interpolation=tvf.InterpolationMode.BICUBIC),              
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
            for i in db_indices.tolist():
                pred_db_paths = database_dataset.img_path_list[i]
                f.write(f'{pred_query_path} {pred_db_paths}\n')


def visualize(top_k_matches: np.ndarray,
              query_dataset: BaseDataset,
              database_dataset: BaseDataset,
              visual_dir: str = './LOGS/visualize',
              img_resize_size: Tuple = (400, 300)) -> None:
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    for q_idx, db_idx in enumerate(tqdm(top_k_matches, ncols=100, desc='Visualizing matches')):
        pred_q_path = query_dataset.img_path_list[q_idx]
        q_array = cv2.imread(pred_q_path, cv2.IMREAD_COLOR)
        q_array = cv2.resize(q_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
        query_label = re.findall("\d+", pred_q_path)[-1]
        query_label = int(query_label.lstrip('0'))
        gap_array = np.ones((q_array.shape[0], 10, 3)) * 255  # white gap
        text = 'R'
        org = (180, 140)  # 文字左下角的起始位置 (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 3  # 字体缩放比例
        color = (0, 0, 255)  # 红色，以 (B, G, R) 格式指定颜色
        thickness = 5  # 文字线条粗细
        for i in db_idx.tolist():
            pred_db_paths = database_dataset.img_path_list[i]
            db_array = cv2.imread(pred_db_paths, cv2.IMREAD_COLOR)
            db_array = cv2.resize(db_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
            ref_label = re.findall("\d+", pred_db_paths)[-1]
            ref_label = int(ref_label.lstrip('0'))
            if(abs(query_label-ref_label) <= 1):
                cv2.putText(db_array, text, org, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            q_array = np.concatenate((q_array, gap_array, db_array), axis=1)

        result_array = q_array.astype(np.uint8)
        # result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        # save result as image using cv2
        cv2.imwrite(f'{visual_dir}/{os.path.basename(pred_q_path)}', result_array)

import re
def calculate_recalls(top_k_matches, query_dataset, database_dataset):
    n_values = [1,5,10]
    top_k = [1,5,10]
    recalls = {}
    for h, j in enumerate(n_values):
        correct_at_n = 0
        for i in range(len(top_k_matches)):
            query_label = re.findall("\d+", query_dataset.img_path_list[i])[-1]
            query_label = int(query_label.lstrip('0'))
            for k in range(j):
                ref_label = re.findall("\d+", database_dataset.img_path_list[top_k_matches[i][k]])[-1]
                ref_label = int(ref_label.lstrip('0'))
                if(abs(query_label-ref_label) <=1):
                    correct_at_n += 1
                    break
        recall_at_n = correct_at_n / len(top_k_matches)
        recalls[j] = recall_at_n
        print("====>Recall@top-{}: {:.4f}".format(top_k[h], recall_at_n))

    return recalls
def main():
    # load images
    query_path = '/media/robot/CBBPS9/dataset/VPAir/queries/'         # path to query images folder path
    datasets_path = '/media/robot/CBBPS9/dataset/VPAir/reference_views/'      # path to database images folder path

    query_dataset = BaseDataset(query_path)
    database_dataset = BaseDataset(datasets_path)

    # load model
    model = load_model('./runPath/e2resnet50_c8_se2gem_32/checkpoints/model_best.pth.tar') 
    
    # set up inference pipeline
    database_pipeline = InferencePipeline(model=model, dataset=database_dataset, feature_dim=256)
    query_pipeline = InferencePipeline(model=model, dataset=query_dataset, feature_dim=256)

    # run inference
    db_global_descriptors = database_pipeline.run(split='db')  # shape: (num_db, feature_dim)
    query_global_descriptors = query_pipeline.run(split='query')  # shape: (num_query, feature_dim)

    # calculate top-k matches
    top_k_matches = calculate_top_k(q_matrix=query_global_descriptors, db_matrix=db_global_descriptors, top_k=10)

    # record query_database_matches
    record_matches(top_k_matches, query_dataset, database_dataset, out_file='./LOGS/record.txt')

    calculate_recalls(top_k_matches,query_dataset, database_dataset)

    # visualize top-k matches
    # visualize(top_k_matches, query_dataset, database_dataset, visual_dir='./LOGS/visualize')


if __name__ == '__main__':
    main()
