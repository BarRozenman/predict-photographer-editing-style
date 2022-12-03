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
import torchvision
from PIL import Image
import torch
from natsort import natsorted
from personal_utils.flags import flags
from personal_utils.plot_utils import scatter_clustering_with_gt_labels_in_2d, scatter_clustering_with_gt_labels_in_3d
from sklearn.cluster import KMeans
from torch import max_pool2d, conv_transpose2d
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
from torchvision.models import vgg
from torchvision.transforms import transforms as T, Resize
from torchvision.transforms import transforms
import seaborn as sns
# Loss function
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Define the Convolutional Autoencoder
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]


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

        self.conv1 = nn.Conv2d(3, 24, 3, padding=1)
        # pooling layer
        self.conv2 = nn.Conv2d(24, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 4, 3, padding=1)
        self.conv4 = nn.Conv2d(4, 8, 3, padding=1)
        # self.conv4 = nn.Conv2d(20, 4, 3, padding=1)
        # pooling layer

        self.lin = nn.Linear(1568, 16)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(1, 12, 4, stride=4)
        self.t_conv2 = nn.ConvTranspose2d(12, 6, 4, stride=4)
        self.t_conv3 = nn.ConvTranspose2d(6, 6, 4, stride=2)  # dsfdsfdfs
        self.t_conv4 = nn.ConvTranspose2d(6, 3, 4, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.conv3(x)
        # x = self.pool(x)
        # x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        embedding = self.lin(x)
        x = embedding.view(embedding.shape[0], 1, 4, 4)
        # get x layer from convnet and return it
        # emb = F.avg_pool2d(x, kernel_size=56, stride=1, padding=0)
        x = F.relu(x)
        # x = self.pool(x)
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
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = t(img)
    return img


def get_contrast(img=None, img_path=None):
    # read image
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = np.array(transforms.ToPILImage()(img))
    # convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # separate channels
    L, A, B = cv2.split(lab)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L, kernel, iterations=1)  # setting the minimal value in the vicinity
    max = cv2.dilate(L, kernel, iterations=1)

    # convert min and max to floats
    min = min.astype(np.float64)
    max = max.astype(np.float64)

    # compute local contrast
    contrast = (max - min) / (max + min)

    # get average across whole image
    average_contrast = 100 * np.mean(contrast)

    print(str(average_contrast) + "%")
    if average_contrast == torch.nan:
        a = 1
    return [average_contrast]


def get_mean_var_rgb(img=None, img_path=None):
    if img_path:
        pil_img = Image.open(img_path).convert('RGB')

    else:
        pil_img = transforms.ToPILImage()(img).convert('RGB')

    img_arr = np.array(pil_img)
    output = 6 * [None]
    for channel_idx in range(img_arr.shape[2]):
        curr_arr = img_arr[..., channel_idx]
        output[2 * channel_idx] = np.sum(curr_arr) / curr_arr.size
        output[2 * channel_idx + 1] = np.std(curr_arr) / (curr_arr.size / 1000000)
    return output


def get_global_features(img=None, img_path=None):
    feat = get_contrast(img)
    rgb_feat = get_mean_var_rgb(img)
    feat.extend(rgb_feat)
    return feat


# create custom pytorch dataset

def rgb_to_lab(srgb):
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]).type(torch.FloatTensor).to(device)

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels,
                                      torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).type(torch.FloatTensor).to(
                                          device))

    epsilon = 6.0 / 29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).type(torch.FloatTensor).to(device)

    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).type(torch.FloatTensor).to(device)

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4.0 / 29.0) * linear_mask + (
            (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ]).type(torch.FloatTensor).to(device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(
        device)
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)


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
        # T.ToPILImage()(image).show()
        # feat = get_contrast(image)
        # if np.isnan(feat[0]):
        #     a=1
        # feat = get_contrast(image)
        # rgb_feat = get_mean_var_rgb(image)
        # feat.extend(rgb_feat)
        return image, img_path  # , feat


def lab_to_rgb(lab):
    lab_pixels = torch.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ]).type(torch.FloatTensor).to(device)
    fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device),
                             lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)

    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
            (fxfyfz_pixels + 0.000001) ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ]).type(torch.FloatTensor).to(device)

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
            ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


def batch_get_contrast(images):
    lab_imgs = (rgb_to_lab(images)[:, 0, :, :])
    max_img = F.max_pool2d(lab_imgs, 3, stride=1, padding=1)
    min_img = -F.max_pool2d(-lab_imgs, 3, stride=1, padding=1)
    contrast = (max_img - min_img) / (max_img + min_img)
    contrast[(contrast < -100) + (contrast > 100)] = torch.mean(contrast[(contrast > -100) * (contrast < 100)])
    # get average across whole image
    average_contrast = torch.mean(contrast, dim=(1, 2))
    return average_contrast


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
    n_epochs = 10
    batch_size = 32
    flags.use_cache = True
    save_embedding = True
    device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.L1Loss()
    dataset = CustomDataset('/home/bar/projects/personal/imagen/data/images', transform)
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = random_split(dataset, [train_sample_num, test_sample_num], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    loss_func = nn.MSELoss()
    global_step = 0
    lambda1 = lambda epoch: 0.65 ** epoch
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

            contrast_loss = loss_func(input_contrast.type(torch.float), output_contrast.type(torch.float)) / 10000
            mean_rgb_loss = loss_func(input_mean_rgb, output_mean_rgb) * 10
            reconstruction_loss = criterion(outputs, images)
            loss = contrast_loss + mean_rgb_loss + reconstruction_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            global_step = previous_global_step + batch_idx + (epoch - 1) * len(train_loader)
            writer.add_scalar('contrast_loss', contrast_loss, global_step)
            writer.add_scalar('mean_rgb_loss', mean_rgb_loss, global_step)
            writer.add_scalar('reconstruction_loss', reconstruction_loss, global_step)
            writer.add_scalar('loss', loss, global_step)
            train_loss += loss.item() * images.size(0)
            if save_embedding:
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
            if batch_idx % 50 == 0:
                # save to png image first input and out images of the batch
                img_grid_input = torchvision.utils.make_grid(images[:4,...])
                img_grid_output = torchvision.utils.make_grid(outputs[:4,...])
                writer.add_image('images input', img_grid_input, global_step)
                writer.add_image('images prediction', img_grid_output, global_step)


                print('Epoch: {} \tBatch: {} \tLoss: {:.6f}'.format(epoch, batch_idx, loss.item()))
                # save model weights
                if batch_idx % 100 != 0:
                    continue
                print('Saving model weights')
                Path(f'./model_weights/{flags.timestamp}').mkdir(parents=True, exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()},
                           f'./model_weights/{flags.timestamp}/conv_autoencoder_epoch_{epoch}_step_{global_step}_{flags.timestamp}.pt')
        scheduler.step()  # step to the next learning rate each epoch
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    writer.close()

    # arg_class = collections.namedtuple('arg',
    #                                    ['data_dir'])
    # data_dir = '/home/bar/projects/personal/imagen/data/images'
    # args = arg_class(data_dir)
    # args.data_dir
    # df = pd.DataFrame(columns=['image_path', 'contrast', 'mean_r', 'var_r', 'mean_g', 'var_g', 'mean_b', 'var_b'])
    # for image in glob(os.path.join(args.data_dir, "*.jpg"))[:100]:
    #     try:
    #         feat = get_contrast(image)
    #     except:  # skip back images
    #         continue
    #     rgb_feat = get_mean_var_rgb(image)
    #     feat.extend(rgb_feat)
    #     feat.insert(0, image)
    #     df.loc[len(df)] = feat
    # df.dropna(inplace=True)
    #
    # img_path = '/home/bar/projects/personal/imagen/data/wedding_content.jpg'
    # contrast = get_contrast(img_path)
    # pil_img = Image.open(img_path)
    # get_mean_var_rgb(img_path)
    # exifdata = pil_img.getexif()
    # photographer

    # that the loss is the 5 values autoencoder of the input vs output

    EMBED_DIMENSION = 5
    EMBED_MAX_NORM = 1

    # class CBOW_Model(nn.Module):
    #     def __init__(self, vocab_size: int):
    #         super(CBOW_Model, self).__init__()
    #         self.embeddings = nn.Embedding(
    #             num_embeddings=vocab_size,
    #             embedding_dim=EMBED_DIMENSION,
    #             max_norm=EMBED_MAX_NORM,
    #         )
    #         self.linear = nn.Linear(
    #             in_features=EMBED_DIMENSION,
    #             out_features=vocab_size,
    #         )
    #
    #     def forward(self, inputs_):
    #         x = self.embeddings(inputs_)
    #         x = x.mean(axis=1)
    #         x = self.linear(x)
    #         return x
    #
    #
    # # from torchtext.vocab import build_vocab_from_iterator
    #
    # MIN_WORD_FREQUENCY = 50
    #
    #
    # def build_vocab(data_iter, tokenizer):
    #     vocab = build_vocab_from_iterator(
    #         map(tokenizer, data_iter),
    #         specials=["<unk>"],
    #         min_freq=MIN_WORD_FREQUENCY,
    #     )
    #     vocab.set_default_index(vocab["<unk>"])
    #     return vocab
    #
    #
    # CBOW_N_WORDS = 4
    # MAX_SEQUENCE_LENGTH = 256
    #
    #
    # def collate_cbow(batch, text_pipeline):
    #     batch_input, batch_output = [], []
    #     for text in batch:
    #         text_tokens_ids = text_pipeline(text)
    #         if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
    #             continue
    #         if MAX_SEQUENCE_LENGTH:
    #             text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
    #         for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
    #             token_id_sequence = text_tokens_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
    #             output = token_id_sequence.pop(CBOW_N_WORDS)
    #             input_ = token_id_sequence
    #             batch_input.append(input_)
    #             batch_output.append(output)
    #
    #     batch_input = torch.tensor(batch_input, dtype=torch.long)
    #     batch_output = torch.tensor(batch_output, dtype=torch.long)
    #     return batch_input, batch_output
    #
    #
    # from torch.utils.data import DataLoader
    # from functools import partial
    #
    # dataloader = DataLoader(
    #     data_iter,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=partial(collate_cbow, text_pipeline=text_pipeline),
    # )
    #
    # embeddings = list(model.parameters())[0]
    # vocab.get_itos()
