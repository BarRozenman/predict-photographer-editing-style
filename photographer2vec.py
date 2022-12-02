import collections
import os
from collections import namedtuple
from glob import glob

import pandas as pd
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
from torchvision.models import vgg
from torchvision.transforms import transforms as T
from torchvision.transforms import transforms

# Loss function
import torch.nn.functional as F


# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x


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

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]).type(torch.FloatTensor).cuda()

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels,
                                      torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).type(torch.FloatTensor).cuda())

    epsilon = 6.0 / 29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).type(torch.FloatTensor).cuda()

    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).type(torch.FloatTensor).cuda()

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4.0 / 29.0) * linear_mask + (
            (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ]).type(torch.FloatTensor).cuda()
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)


class CustomDataset(ImageFolder):
    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        feat = get_contrast(image)
        rgb_feat = get_mean_var_rgb(image)
        feat.extend(rgb_feat)
        return image, img_path, feat


def lab_to_rgb(lab):
    lab_pixels = torch.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ]).type(torch.FloatTensor).cuda()
    fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda(), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).cuda()
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).cuda()

    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
            (fxfyfz_pixels + 0.000001) ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).cuda())

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ]).type(torch.FloatTensor).cuda()

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).cuda()
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).cuda()
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
            ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


if __name__ == '__main__':

    # Instantiate the model
    model = ConvAutoencoder().requires_grad_(True)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Epochs
    n_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    criterion = nn.BCELoss()
    dataset = CustomDataset('/home/bar/projects/personal/imagen/data/images', transform)
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_sample_num, test_sample_num])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=12, num_workers=0)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=12, num_workers=0)
    loss_func = nn.MSELoss()
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        # Training
        for data in train_loader:
            images, img_path_image, features = data
            features = torch.stack(features, 1)
            features = features[:, 1::2]
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if False:
                pred_feat = []
                for i in outputs:
                    curr_pred_feat = torch.mean(outputs, dim=(2, 3))
                    # curr_pred_feat = get_global_features(i)
                    pred_feat.append(torch.Tensor(curr_pred_feat))
                pred_feat = torch.stack(pred_feat)
            pred_feat = torch.mean(outputs, dim=(2, 3))
            max_img = F.max_pool2d(images, 3, stride=1, padding=1)
            min_img = -F.max_pool2d(-images, 3, stride=1, padding=1)

            T.ToPILImage()(o[0]).show()

            loss = loss_func(features.type(torch.float), pred_feat.type(torch.float))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    arg_class = collections.namedtuple('arg',
                                       ['data_dir'])
    data_dir = '/home/bar/projects/personal/imagen/data/images'
    args = arg_class(data_dir)
    args.data_dir
    df = pd.DataFrame(columns=['image_path', 'contrast', 'mean_r', 'var_r', 'mean_g', 'var_g', 'mean_b', 'var_b'])
    for image in glob(os.path.join(args.data_dir, "*.jpg"))[:100]:
        try:
            feat = get_contrast(image)
        except:  # skip back images
            continue
        rgb_feat = get_mean_var_rgb(image)
        feat.extend(rgb_feat)
        feat.insert(0, image)
        df.loc[len(df)] = feat
    df.dropna(inplace=True)

    img_path = '/home/bar/projects/personal/imagen/data/wedding_content.jpg'
    contrast = get_contrast(img_path)
    pil_img = Image.open(img_path)
    get_mean_var_rgb(img_path)
    exifdata = pil_img.getexif()
    # photographer

    # that the loss is the 5 values autoencoder of the input vs output

    EMBED_DIMENSION = 5
    EMBED_MAX_NORM = 1


    class CBOW_Model(nn.Module):
        def __init__(self, vocab_size: int):
            super(CBOW_Model, self).__init__()
            self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=EMBED_DIMENSION,
                max_norm=EMBED_MAX_NORM,
            )
            self.linear = nn.Linear(
                in_features=EMBED_DIMENSION,
                out_features=vocab_size,
            )

        def forward(self, inputs_):
            x = self.embeddings(inputs_)
            x = x.mean(axis=1)
            x = self.linear(x)
            return x


    # from torchtext.vocab import build_vocab_from_iterator

    MIN_WORD_FREQUENCY = 50


    def build_vocab(data_iter, tokenizer):
        vocab = build_vocab_from_iterator(
            map(tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=MIN_WORD_FREQUENCY,
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab


    CBOW_N_WORDS = 4
    MAX_SEQUENCE_LENGTH = 256


    def collate_cbow(batch, text_pipeline):
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)
            if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
                continue
            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
            for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
                output = token_id_sequence.pop(CBOW_N_WORDS)
                input_ = token_id_sequence
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output


    from torch.utils.data import DataLoader
    from functools import partial

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline),
    )

    embeddings = list(model.parameters())[0]
    vocab.get_itos()
