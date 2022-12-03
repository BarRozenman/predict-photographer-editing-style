import collections
import os
from collections import namedtuple
from glob import glob
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torch
from natsort import natsorted
from personal_utils.flags import flags
from personal_utils.plot_utils import scatter_clustering_with_gt_labels_in_2d,scatter_multiple_images, scatter_clustering_with_gt_labels_in_3d
from sklearn.cluster import KMeans
from torch import max_pool2d, conv_transpose2d
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Dataset
from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
from torchvision.models import vgg
from torchvision.transforms import transforms as T, Resize
from torchvision.transforms import transforms
import seaborn as sns
# Loss function
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def create_user_initial_vector(num_users):
    # Create a vector of random with size 3 by num_photographers to represent the photographer
    return torch.rand(num_users, 3).requires_grad_(True)


class MixedNetwork(nn.Module):
    def __init__(self):
        super(MixedNetwork, self).__init__()

        image_modules = list(models.resnet50().children())[:-1]
        self.image_features = nn.Sequential(*image_modules)

        self.landmark_features = nn.Sequential(
            nn.Linear(in_features=96, out_features=192, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=192, out_features=1000, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25))

        self.combined_features = nn.Sequential(
            # change this input nodes
            nn.Linear(3048, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, image, landmarks):
        a = self.image_features(image)
        b = self.landmark_features(landmarks)
        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1)), dim=1)
        x = self.combined_features(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    images_editing_features = pd.read_csv('/home/bar/projects/personal/imagen/images_editing_features.csv')
    images_embeddings = pd.read_csv('/home/bar/projects/personal/imagen/embeddings/embedding_epoch_2022-12-03_0928.csv')
    # images_embeddings.join(images_editing_features, on='image_path')
    df =images_editing_features[['path','photographer']]
    df = df.set_index('path')
    sr  = df['photographer']

    images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    # images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    images_embeddings.dropna(inplace=True)
    images_embeddings['photographer'] = images_embeddings['photographer'].astype('int')

    nn  = images_embeddings.iloc[:,1:-1]
    im_emb=images_embeddings.set_index('image_path')
    common_paths = np.intersect1d(images_editing_features['path'].values,im_emb.index.values)
    im_emb.loc[common_paths]
    news_df = im_emb.groupby(level=0)

    if flags.debug:
        scatter_clustering_with_gt_labels_in_2d(images_editing_features.iloc[:,1:-1])
        scatter_clustering_with_gt_labels_in_2d(images_embeddings.iloc[:,1:],title='image embedding space')
        x =images_editing_features.iloc[:, 1]
        y =images_editing_features.iloc[:, 2]
        scatter_multiple_images(x, y, images_editing_features.iloc[:, 0], shown_amount=200,zoom=0.08)
        plt.show()
    num_users = len(np.unique(images_editing_features['photographer']))
    photo_embed = create_user_initial_vector(num_users)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_files = glob('embeddings/*.csv')
    files = glob('model_weights/**/edit_predictor_*.pt')


    class CustomDataset(Dataset):
        def __init__(self,images_embeddings,images_editing_features):
            super(CustomDataset, self).__init__()
            self.imgs = images_embeddings[['image_path', 'photographer']].values
            self.image_embd = images_embeddings.iloc[:, 1:-1].values
            self.images_editing_features = images_editing_features.iloc[:, 1:].values
            self.labels = images_embeddings['photographer'].values
        def __getitem__(self, idx):
            img_path = self.imgs[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # T.ToPILImage()(image).show()
            # feat = get_contrast(image)
            # if np.isnan(feat[0]):
            #     a=1
            # feat = get_contrast(image)
            # rgb_feat = get_mean_var_rgb(image)
            # feat.extend(rgb_feat)
            return image, img_path  # , feat


    lr = 0.01
    lambda1 = lambda epoch: 0.95 ** epoch
    if len(files) > 0 and flags.use_cache:
        last_run = natsorted(files)[-1]
        epoch_num_from_file = int(Path(files[-1]).name.split('_')[3])
        previous_global_step = int(Path(files[-1]).name.split('_')[5])
        previous_epoch_idx = epoch_num_from_file
        model = MixedNetwork().to(device)
        loaded_data = torch.load(last_run)
        model.load_state_dict(loaded_data['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(loaded_data['optimizer'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        scheduler.load_state_dict(loaded_data['scheduler'])
    else:
        previous_epoch_idx = 0
        previous_global_step = 0
        model = MixedNetwork()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

