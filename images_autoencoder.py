import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from glob import glob
from natsort import natsorted
from pathlib import Path
from personal_utils.flags import flags
from personal_utils.plot_utils import scatter_clustering_with_gt_labels_in_2d, scatter_clustering_with_gt_labels_in_3d
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import transforms as T, Resize
from tqdm import tqdm


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(  # 784
            nn.Conv2d(3, 32, stride=2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=5, padding=2),
            nn.Flatten(),
            nn.Linear(12544, 10)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(10, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(3, 3, stride=2, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(3, 3, stride=2, kernel_size=(3, 3), padding=1),
            # Trim(),  # 1x29x29 -> 1x28x28
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        x = Resize(224)(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # pooling layer
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 6, 3, padding=1)
        self.conv4 = nn.Conv2d(6, 8, 3, padding=1)
        self.lin = nn.Linear(1568, 16)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(1, 12, 4, stride=4)
        self.t_conv2 = nn.ConvTranspose2d(12, 6, 4, stride=4)
        self.t_conv3 = nn.ConvTranspose2d(6, 6, 4, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(6, 3, 4, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        embedding = self.lin(x)
        x = embedding.view(embedding.shape[0], 1, 4, 4)
        x = F.relu(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))

        x = torch.sigmoid(self.t_conv4(
            x))  # since after relu everything is positive, we need to use sigmoid to get the values between 0 and 1,
        # can be change to tanh if we want to get values between -1 and 1 (by removing the sigmoid)
        x = Resize(224)(x)
        if flags.debug:
            T.ToPILImage()(x[0, ...]).show()
        return x, embedding


def transform(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = t(img)
    return img


def get_global_features(img=None, img_path=None):
    feat = get_contrast(img)
    rgb_feat = get_mean_var_rgb(img)
    feat.extend(rgb_feat)
    return feat


def create_image_features_df():
    df = pd.DataFrame(columns=['path', 'contrast', 'mean_r', 'mean_g', 'mean_b'])
    for i in range(len(img_path_image)):
        df = df.append({'path': img_path_image[i], 'contrast': input_contrast[i].item(),
                        'mean_r': input_mean_rgb[i][0].item(), 'mean_g': input_mean_rgb[i][1].item(),
                        'mean_b': input_mean_rgb[i][2].item()}, ignore_index=True)

    df.to_csv('images_editing_features.csv', index=False)
    split_images_to_photographers()


class CustomDataset(ImageFolder):
    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # , feat


def batch_get_mean_rgb(images):
    mean_rgb = torch.mean(images, dim=(2, 3))
    return mean_rgb


def split_images_to_photographers():
    df = pd.read_csv('images_editing_features.csv')
    clusterer = KMeans(n_clusters=30, random_state=0).fit(df[['contrast', 'mean_r', 'mean_g', 'mean_b']])
    scatter_clustering_with_gt_labels_in_2d(df[['contrast', 'mean_r', 'mean_g', 'mean_b']], clusterer.labels_)
    scatter_clustering_with_gt_labels_in_3d(df[['contrast', 'mean_r', 'mean_g', 'mean_b']], clusterer.labels_)
    plt.show()
    df['photographer'] = clusterer.labels_
    df.to_csv('images_editing_features.csv', index=False)


if __name__ == '__main__':
    flags.debug = False
    writer = SummaryWriter(f'runs/{flags.timestamp}')
    num_workers = 12
    n_epochs = 30
    batch_size = 64
    flags.use_cache = False
    save_embedding = True
    device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.L1Loss()
    dataset = CustomDataset('/home/bar/projects/personal/imagen/data/images', transform)
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = random_split(dataset, [train_sample_num, test_sample_num],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    loss_func = nn.MSELoss()
    global_step = 0
    lambda1 = lambda epoch: 0.999 ** np.ceil(epoch // 4)
    files = glob('model_weights/**/conv_autoencoder_*.pt')

    epoch_num_from_file = 0
    batch_num_from_file = 0
    embedding_df = pd.DataFrame(columns=['image_path'] + [str(i) for i in range(16)])

    lr = 0.01
    if len(files) > 0 and flags.use_cache:
        last_run = natsorted(files)[-1]
        epoch_num_from_file = int(Path(files[-1]).name.split('_')[3])
        previous_global_step = int(Path(files[-1]).name.split('_')[5])
        previous_epoch_idx = epoch_num_from_file
        model = ConvAutoencoder().to(device)
        loaded_data = torch.load(last_run)
        model.load_state_dict(loaded_data['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(loaded_data['optimizer'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        scheduler.load_state_dict(loaded_data['scheduler'])
    else:
        previous_epoch_idx = 0
        previous_global_step = 0
        model = ConvAutoencoder()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # save image features to dataframe
    for epoch in range(1 + previous_epoch_idx, n_epochs + 1 + previous_epoch_idx):
        # monitor training loss
        train_loss = 0.0
        # Training
        for batch_idx, batch in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            images, img_path_image = batch
            images = images.to(device)
            if flags.debug:
                T.ToPILImage()(images[0, ...]).show()
            # ===================forward=====================
            outputs, emb = model(images)
            if flags.debug:
                T.ToPILImage()(outputs[0, ...]).show()
            input_contrast = batch_get_contrast(images)
            output_contrast = batch_get_contrast(outputs)
            input_mean_rgb = batch_get_mean_rgb(images)
            output_mean_rgb = batch_get_mean_rgb(outputs)
            input_contrast[torch.isinf(input_contrast)] = input_contrast[~torch.isinf(input_contrast)].mean()

            contrast_loss = loss_func(input_contrast.type(torch.float), output_contrast.type(torch.float)) / 100000
            mean_rgb_loss = loss_func(input_mean_rgb, output_mean_rgb) * 10
            reconstruction_loss = criterion(outputs, images)
            loss = contrast_loss + mean_rgb_loss + reconstruction_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # step to the next learning rate each batch -> faster

            # ===================log========================
            global_step = previous_global_step + batch_idx + (epoch - 1) * len(train_loader)
            writer.add_scalar('contrast_loss', contrast_loss, global_step)
            writer.add_scalar('mean_rgb_loss', mean_rgb_loss, global_step)
            writer.add_scalar('reconstruction_loss', reconstruction_loss, global_step)
            writer.add_scalar('loss', loss, global_step)
            train_loss += loss.item() * images.size(0)
            if save_embedding:
                print(scheduler.get_last_lr())
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)

                embedding_file_path = f'embeddings/embedding_epoch_{epoch}_{flags.timestamp}.csv'
                embedding = emb.detach().cpu().numpy()
                for i in range(len(img_path_image)):
                    new_row = {'image_path': img_path_image[i]}
                    new_row.update({str(feat_idx): embedding[i][feat_idx] for feat_idx in range(len(embedding[i]))})
                    embedding_df.loc[len(embedding_df), :] = new_row
                if Path(embedding_file_path).exists():
                    embedding_df.to_csv(embedding_file_path, mode='a', index=False, header=False)
                else:
                    embedding_df.to_csv(embedding_file_path, index=False)
                embedding_df = pd.DataFrame(columns=['image_path'] + [str(i) for i in range(16)])
            if batch_idx % 20 == 0:
                # save to png image first input and out images of the batch
                example_image_idx = np.random.choice(np.arange(images.shape[0]), 4, replace=False)

                img_grid_input = torchvision.utils.make_grid(images[example_image_idx, ...])
                img_grid_output = torchvision.utils.make_grid(outputs[example_image_idx, ...])
                writer.add_image('images input', img_grid_input, global_step)
                writer.add_image('images prediction', img_grid_output, global_step)

                print('Epoch: {} \tBatch: {} \tLoss: {:.6f}'.format(epoch, batch_idx, loss.item()))
                # save model weights
                if batch_idx % 40 != 0:
                    continue
                print('Saving model weights')
                Path(f'./model_weights/{flags.timestamp}').mkdir(parents=True, exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()},
                           f'./model_weights/{flags.timestamp}/conv_autoencoder{flags.timestamp}_epoch_{epoch}_step_{global_step}_.pt')
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    writer.close()
