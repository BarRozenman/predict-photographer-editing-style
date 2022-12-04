import collections
import os
import sys
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
from torch import max_pool2d, conv_transpose2d, Tensor
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
    return torch.rand(num_users, 3, generator=torch.Generator().manual_seed(42))  # .requires_grad_(True)


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
    def __init__(self, norm_params=None, train=True):
        super(MixedNetwork, self).__init__()
        self.train = train
        self.norm_params = norm_params
        # image_modules = list(models.resnet50().children())[:-1]
        # self.image_features = nn.Sequential(*image_modules)
        # self.user_features = user_feaktures
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
            # nn.ReLU())

    def norm_test_input(self, img_features, user_features):

        img_features = (img_features - self.norm_params['images_embeddings_min']) / (
                self.norm_params['images_embeddings_max'] - self.norm_params['images_embeddings_min'])
        return img_features, user_features

    def forward(self, img_features, user_features):
        # user_features = user_features.requires_grad_(True)
        if self.train:
            pass
        else:
            img_features, user_features = self.norm_test_input(img_features, user_features)
        a = self.img_features(img_features.float())
        b = self.user_features(user_features.float())
        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1)), dim=1)
        x = self.combined_features(x)
        x = torch.sigmoid(x)

        return x


def plot_user_temporal_embedding(users_embed_temporal, loss_for_plot, sink_path):
    # Plot the temporal embedding of the users
    last_users_embed_temporal = pd.DataFrame(users_embed_temporal[-1, :, :], columns=['x', 'y', 'z'])

    dim_reduction_func = PCA(n_components=2).fit(last_users_embed_temporal)
    for i in range(users_embed_temporal.shape[0]):
        curr_users_embed = users_embed_temporal[i, ...].cpu().detach().numpy()
        dim_2_users = pd.DataFrame(dim_reduction_func.transform(curr_users_embed), columns=['x', 'y'])

        ax = sns.scatterplot(data=dim_2_users, x='x', y='y', hue=dim_2_users.index.values, palette="deep")
        # remove legend
        ax.get_legend().remove()
        # limit x and y axis to -1 to 1
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        plt.title(f'Users embedding at step {str(i).zfill(4)}\n Loss: {round(loss_for_plot[i], 5):8.3f}')
        plt.savefig(append2file_name(sink_path, str(i)))
        plt.close('all')
    crawl_images_from_users_embedding_frames(flags.timestamp)


def crawl_images_from_users_embedding_frames(curr_time):
    from personal_utils.video_utils import generate_video_from_frames_paths
    all_frames = natsorted(glob('visuals/user_embed_temporal_*.png'))
    frames = []
    for f in all_frames:
        if curr_time in f:
            frames.append(f)
    generate_video_from_frames_paths(frames, f'visuals/user_embed_temporal_{flags.timestamp}.mp4', fps=8)


def load_model(model_path=None):
    dir = f'model_weights/user_predictor'
    last_file = natsorted(glob(f'{dir}/**/*.pt'))[-1]
    model = torch.load(last_file)

    # model.to(device)
    return model


def use_model_to_predict(model, dataset):
    user_features_file = natsorted(glob('user_embedding/users_embed_temporal_*.pt'))[-1]
    user_embed:Tensor = torch.load(user_features_file)
    model.train = False
    user_embed.repeat(batch_size, 1)
    users_id =np.random.choice(range(30),2, replace=False)
    user_embed_batch = user_embed[users_id[0], ...].repeat(batch_size, 1)
    user_embed_batch_2 = user_embed[users_id[1], ...].repeat(batch_size, 1)
    output_all = None
    with torch.no_grad():
        for i, data in enumerate(dataset):
            img_features = data['image_embed']
            # img_features = img_features.to(device)
            # user_features = user_features.to(device)
            output = model(img_features, user_embed_batch)
            output2 = model(img_features, user_embed_batch_2)
            if output_all is None:
                output_all = output.unsqueeze(0)
            else:
                output_all = torch.cat((output_all,output.unsqueeze(0)), dim=0)
            if i == 10:
                break
    a=1
    if flags.debug:
        sns.histplot(output[:,0])
        sns.histplot(output2[:,1])
        plt.show()


if __name__ == '__main__':
    images_editing_features = pd.read_csv('/home/bar/projects/personal/imagen/images_editing_features.csv')
    images_embeddings_files = glob('/home/bar/projects/personal/imagen/embeddings/embedding_epoch*.csv')
    # images_embeddings = pd.read_csv(natsorted(images_embeddings_files)[-1])
    images_embeddings = pd.read_csv(
        '/home/bar/projects/personal/imagen/embeddings/embedding_epoch_10_2022-12-03_1851.csv')
    # images_embeddings.join(images_editing_features, on='image_path')
    df = images_editing_features[['path', 'photographer']]
    df = df.set_index('path')
    sr = df['photographer']
    images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    # images_embeddings = images_embeddings.join(sr, on='image_path', how='left')
    images_embeddings.dropna(inplace=True)
    images_embeddings['photographer'] = images_embeddings['photographer'].astype('int')

    im_emb = images_embeddings.set_index('image_path')
    common_paths = np.intersect1d(images_editing_features['path'].values, im_emb.index.values)
    im_emb = im_emb.loc[common_paths]
    images_editing_features.set_index('path', inplace=True)
    images_editing_features = images_editing_features.loc[common_paths]
    # normalize the data
    v = images_editing_features.iloc[:, :-1].values
    norm_params = {'images_editing_features_min': v.min(0), 'images_editing_features_max': v.max(0)}
    v = (v - v.min(0)) / (v.max(0) - v.min(0))

    images_editing_features.iloc[:, :-1] = v
    v = im_emb.iloc[:, :-1].values
    norm_params['images_embeddings_min'] = v.min(0)
    norm_params['images_embeddings_max'] = v.max(0)
    v = (v - v.min(0)) / (v.max(0) - v.min(0))
    im_emb.iloc[:, :-1] = v

    # normalize the data
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
    batch_size = 32
    num_workers = 0
    n_epochs = 200
    loss_func = nn.MSELoss()
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = random_split(dataset, [train_sample_num, test_sample_num])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    # use_model_to_predict(load_model(), test_loader)
    lr = 0.005
    writer = SummaryWriter(f'runs/user_embedding/{flags.timestamp}')
    opt_state = None
    lambda1 = lambda epoch: 0.999 ** np.ceil(epoch // 2)

    if len(files) > 0 and flags.use_cache:
        last_run = natsorted(files)[-1]
        epoch_num_from_file = int(Path(files[-1]).name.split('_')[3])
        previous_global_step = int(Path(files[-1]).name.split('_')[5])
        previous_epoch_idx = epoch_num_from_file
        model = MixedNetwork(norm_params=norm_params).to(device)
        loaded_data = torch.load(last_run)
        model.load_state_dict(loaded_data['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(loaded_data['optimizer'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        scheduler.load_state_dict(loaded_data['scheduler'])
    else:
        previous_epoch_idx = 0
        previous_global_step = 0
        model = MixedNetwork(norm_params=norm_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    loss_arr = []
    loss_for_plot = [1]
    for epoch in tqdm(range(1 + previous_epoch_idx, n_epochs + 1 + previous_epoch_idx), total=n_epochs):
        for batch_idx, data in enumerate(train_loader):
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
            pred = model(image_embed, user_embed)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            opt_state = optimizer.state_dict()
            loss_arr.append(loss.item())
            writer.add_scalar('loss', loss, global_step)
            users_embed[user_embed_idx] = user_embed.clone().detach().cpu()
            if global_step % 200 == 0:
                # evaluate the model on the test set
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        image_embed = data['image_embed'].to(device)
                        images_editing_features = data['images_editing_features'].to(device)
                        label = data['images_editing_features'].to(device)
                        user_embed_idx = data['label']
                        user_embed = users_embed[user_embed_idx]
                        user_embed = torch.tensor(user_embed, requires_grad=True, device=device)
                        pred = model(image_embed, user_embed)
                        test_loss += loss_func(pred, label)
                writer.add_scalar('test_loss', test_loss, global_step)

                print('learning rate', scheduler.get_last_lr()[0])
                loss_for_plot.append(np.mean(loss_arr[-1000:]))
                loss_arr = []
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)

                users_embed_temporal = torch.cat((users_embed_temporal, users_embed.unsqueeze(0)), dim=0)
                # save last user embedding
                # save model
                dir = f'model_weights/user_predictor/{flags.timestamp}'
                Path(dir).mkdir(parents=True, exist_ok=True)
                torch.save(model, f'{dir}/edit_predictor_{flags.timestamp}_{epoch}_{global_step}.pt')
                # save user temporal embedding
                # torch.save(users_embed_temporal.cpu(), f'users_embedding/user_embed_temporal_{flags.timestamp}.pt')
                # plot user temporal embedding tensor as heatmap

            # print learning rate
        scheduler.step()
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss.item()))
    torch.save(users_embed_temporal[-1, ...], f'user_embedding/users_embed_temporal_{flags.timestamp}.pt')
    plot_user_temporal_embedding(users_embed_temporal, loss_for_plot,
                                 f'visuals/user_embed_temporal_{flags.timestamp}.png')
