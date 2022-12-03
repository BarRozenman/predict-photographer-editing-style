import collections
import os
from collections import namedtuple
from glob import glob
from pathlib import Path
from itertools import chain

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torch
from natsort import natsorted
from personal_utils.file_utils import append2file_name
from personal_utils.flags import flags
from personal_utils.plot_utils import scatter_clustering_with_gt_labels_in_2d, scatter_multiple_images, \
    scatter_clustering_with_gt_labels_in_3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import max_pool2d, conv_transpose2d
from torch.autograd import Variable
from torch.optim import Adam
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
    return torch.rand(num_users, 3,generator=torch.Generator().manual_seed(42))#.requires_grad_(True)


class CustomDataset(Dataset):
    def __init__(self, im_emb, images_editing_features, users_embed):
        super(CustomDataset, self).__init__()
        self.imgs_paths = im_emb.index.values
        self.image_embed = im_emb.iloc[:, :-1].values
        self.images_editing_features = images_editing_features.iloc[:, :-1].values
        self.labels = im_emb['photographer'].values
        self.users_embed = users_embed

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # self.image_embed[idx], self.images_editing_features[idx], self.labels[idx], self.users_embed[
        #     self.labels[idx]]}

        items_dict = {
            'image_embed': torch.tensor(self.image_embed[idx], dtype=torch.float32),
            'images_editing_features': torch.tensor(self.images_editing_features[idx], dtype=torch.float32),
            'label': self.labels[idx],
            'user_embed': self.users_embed[self.labels[idx]]
        }
        return items_dict


class MixedNetwork(nn.Module):
    def __init__(self):
        super(MixedNetwork, self).__init__()

        # image_modules = list(models.resnet50().children())[:-1]
        # self.image_features = nn.Sequential(*image_modules)
        # self.user_features = user_features
        self.img_features = nn.Sequential(
            nn.Linear(in_features=16, out_features=28, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=28, out_features=28, bias=False),
            nn.Linear(in_features=28, out_features=10, bias=False),
            nn.ReLU(inplace=True))

        self.user_features = nn.Sequential(
            nn.Linear(in_features=3, out_features=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=3, out_features=3, bias=False),
            nn.ReLU(inplace=True))
        # nn.Dropout(p=0.25))

        self.combined_features = nn.Sequential(
            # change this input nodes
            nn.Linear(13, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4))
        nn.ReLU(),

    def forward(self, img_features, user_features):
        # user_features = user_features.requires_grad_(True)
        a = self.img_features(img_features)
        b = self.user_features(user_features)
        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1)), dim=1)
        x = self.combined_features(x)
        x = torch.sigmoid(x)
        return x


def plot_user_temporal_embedding(users_embed_temporal, sink_path):
    # Plot the temporal embedding of the users
    last_users_embed_temporal = pd.DataFrame(users_embed_temporal[-1,:,:], columns=['x', 'y', 'z'])

    dim_reduction_func = PCA(n_components=2).fit(last_users_embed_temporal)
    for i in range(users_embed_temporal.shape[0]):
        curr_users_embed =users_embed_temporal[i, ...].cpu().detach().numpy()
        dim_2_users = pd.DataFrame(dim_reduction_func.transform(curr_users_embed), columns=['x', 'y'])

        ax =sns.scatterplot(data=dim_2_users, x='x', y='y', hue=dim_2_users.index.values, palette="deep")
        # remove legend
        ax.get_legend().remove()
        # limit x and y axis to -1 to 1
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        plt.title(f'User temporal embedding at step {i}')
        plt.savefig(append2file_name(sink_path,str(i)))
        plt.close('all')

def crawl_images_from_usres_embbeding_frames():
    from personal_utils.video_utils import generate_video_from_frames_paths
    frames = natsorted(glob.glob('visuals/user_embed_temporal_*.png'))
    generate_video_from_frames_paths(frames, 'visuals/user_embed_temporal.mp4', fps=1)
if __name__ == '__main__':
    images_editing_features = pd.read_csv('/home/bar/projects/personal/imagen/images_editing_features.csv')
    images_embeddings = pd.read_csv(
        '/home/bar/projects/personal/imagen/embeddings/embedding_epoch_2_2022-12-03_1503.csv')
    # images_embeddings.join(images_editing_features, on='image_path')
    df = images_editing_features[['path', 'photographer']]
    df = df.set_index('path')
    sr = df['photographer']
    images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    # images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    images_embeddings.dropna(inplace=True)
    images_embeddings['photographer'] = images_embeddings['photographer'].astype('int')

    # nn = images_embeddings.iloc[:, 1:-1]
    im_emb = images_embeddings.set_index('image_path')
    common_paths = np.intersect1d(images_editing_features['path'].values, im_emb.index.values)
    im_emb = im_emb.loc[common_paths]
    images_editing_features.set_index('path', inplace=True)
    images_editing_features = images_editing_features.loc[common_paths]
    if flags.debug:
        scatter_clustering_with_gt_labels_in_2d(images_editing_features.iloc[:, 2:-1])
        scatter_clustering_with_gt_labels_in_2d(images_embeddings.iloc[:, 1:], title='image embedding space')
        x = images_editing_features.iloc[:, 1]
        y = images_editing_features.iloc[:, 2]
        scatter_multiple_images(x, y, images_editing_features.iloc[:, 0], shown_amount=200, zoom=0.08)
        plt.show()
    num_users = len(np.unique(images_editing_features['photographer']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    users_embed = create_user_initial_vector(num_users).to(device)
    users_embed_temporal = users_embed.unsqueeze(0)
    embed_files = glob('embeddings/*.csv')
    files = glob('model_weights/**/edit_predictor_*.pt')

    flags.use_cache = False
    dataset = CustomDataset(im_emb, images_editing_features, users_embed)
    batch_size = 2
    num_workers = 0
    n_epochs=20
    loss_func = nn.MSELoss()
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = random_split(dataset, [train_sample_num, test_sample_num])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    lr = 0.001
    writer = SummaryWriter(f'runs/user_embedding/{flags.timestamp}')
    opt_state = None
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
    for epoch in tqdm(range(1 + previous_epoch_idx, n_epochs + 1 + previous_epoch_idx),total=n_epochs):
        for batch_idx,data in enumerate(train_loader):
            global_step = previous_global_step + batch_idx + (epoch - 1) * len(train_loader)
            image_embed = data['image_embed'].to(device)
            images_editing_features = data['images_editing_features'].to(device)
            label = data['images_editing_features'].to(device)
            user_embed_idx = data['label']
            user_embed = users_embed[user_embed_idx]
            user_embed = torch.tensor(user_embed, requires_grad=True, device=device)
            optimizer = torch.optim.Adam(list(model.parameters()) + [user_embed], lr=lr)
            if opt_state is not None:
                pass
                # optimizer.load_state_dict(opt_state)
            pred = model(image_embed, user_embed)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            opt_state = optimizer.state_dict()

            writer.add_scalar('loss', loss, global_step)

            users_embed[user_embed_idx] =user_embed.clone().detach().cpu()
            scheduler.step()
            if global_step % 100 == 0:
                users_embed_temporal = torch.cat((users_embed_temporal, users_embed.unsqueeze(0)), dim=0)
                # save user temporal embedding
                torch.save(users_embed_temporal.cpu(), f'users_embedding/user_embed_temporal_{flags.timestamp}.pt')
                # plot user temporal embedding tensor as heatmap




            scheduler.step()
            #print learning rate
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss.item()))
    plot_user_temporal_embedding(users_embed_temporal, f'visuals/user_embed_temporal_{flags.timestamp}.png')

