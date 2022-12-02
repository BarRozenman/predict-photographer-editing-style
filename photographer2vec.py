import collections
import os
from collections import namedtuple
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torch
from natsort import natsorted
from personal_utils.flags import flags
from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
from torchvision.models import vgg
from torchvision.transforms import transforms as T
from torchvision.transforms import transforms
import seaborn as sns
# Loss function
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 24, 3, padding=1)
        # pooling layer
        self.conv2 = nn.Conv2d(24, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 6, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(6, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        x = torch.sigmoid(self.t_conv3(
            x))  # since after relu everything is positive, we need to use sigmoid to get the values between 0 and 1,
        # can be change to tanh if we want to get values between -1 and 1 (by removing the sigmoid)

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
        df = df.append({'path': img_path_image[i], 'contrast': output_contrast[i].item(),
                        'mean_r': output_mean_rgb[i][0].item(), 'mean_g': output_mean_rgb[i][1].item(),
                        'mean_b': output_mean_rgb[i][2].item()}, ignore_index=True)

    df.to_csv('images_editing_features.csv')


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
    contrast[contrast > 100] = torch.mean(contrast[contrast < 10])
    # q1 = df[col].quantile(0.25)
    # q3 = df[col].quantile(0.75)
    # iqr = q3 - q1
    # lower_fence = q1 - iqr * 1.5
    # upper_fence = q3 + iqr * 1.5

    # Counter(contrast.cpu().numpy().flatten().round(2))
    # print('minmax',max_img + min_img)
    from collections import Counter

    # get average across whole image
    average_contrast = torch.mean(contrast, dim=(1, 2))
    return average_contrast


def outlier(col: str, df_: pd.DataFrame = None, remove: bool = False) -> list:
    """This function calculates the upper and lower fence
    of any column and can also remove from the dataset"""
    q1 = df_[col].quantile(0.25)
    q3 = df_[col].quantile(0.75)

    iqr = q3 - q1
    lower_fence = q1 - iqr * 1.5
    upper_fence = q3 + iqr * 1.5

    if remove:
        temp = df_[(df_[col] > lower_fence) & (df_[col] < upper_fence)]
        return temp

    return [lower_fence, upper_fence]


def batch_get_mean_rgb(images):
    mean_rgb = torch.mean(images, dim=(2, 3))
    return mean_rgb

def  split_images_to_photographers():
    df = pd.read_csv('images_editing_features.csv')
    df['photographer'] = df['image'].apply(lambda x: x.split('/')[0])
    df['image'] = df['image'].apply(lambda x: x.split('/')[1])
    df.to_csv('images_editing_features.csv', index=False)
    df = pd.read_csv('images_editing_features.csv')
if __name__ == '__main__':
    flags.debug = False
    # Instantiate the model
    # Optimizer
    # Epochs
    writer = SummaryWriter(f'runs/{flags.timestamp}')
    num_workers = 0
    n_epochs = 2
    device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.L1Loss()
    dataset = CustomDataset('/home/bar/projects/personal/imagen/data/images', transform)
    train_sample_num = int(len(dataset) * 0.7)
    test_sample_num = len(dataset) - int(len(dataset) * 0.7)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_sample_num, test_sample_num])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=12, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=12, num_workers=num_workers)
    loss_func = nn.MSELoss()
    global_step = 0
    lambda1 = lambda epoch: 0.95 ** epoch
    files = glob('model_weights/**/conv_autoencoder_*.pt')
    epoch_num_from_file = 0
    batch_num_from_file = 0
    lr = 0.001
    if len(files) > 0:
        last_run = natsorted(files)[-1]
        epoch_num_from_file = int(Path(files[-1]).name.split('_')[3])
        previous_epoch_idx = epoch_num_from_file
        # batch_num_from_file = int(last_run.split('_')[5].split('.')[0])
        model = ConvAutoencoder().to(device)
        loaded_data = torch.load(last_run)
        model.load_state_dict(loaded_data['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(loaded_data['optimizer'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        scheduler.load_state_dict(loaded_data['scheduler'])
    else:
        previous_epoch_idx = 0
        model = ConvAutoencoder()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # save image features to dataframe
    for epoch in range(1 + previous_epoch_idx, n_epochs + 1 + previous_epoch_idx):
        scheduler.step()
        # monitor training loss
        train_loss = 0.0
        # Training
        # df = pd.DataFrame(columns=['path', 'contrast', 'mean_r', 'mean_g', 'mean_b'])

        for batch_idx, batch in tqdm(enumerate(train_loader, 1)):
            # for i in range(len(img_path_image)):
            #     df = df.append({'path': img_path_image[i], 'contrast': output_contrast[i].item(),
            #                     'mean_r': output_mean_rgb[i][0].item(), 'mean_g': output_mean_rgb[i][1].item(),
            #                     'mean_b': output_mean_rgb[i][2].item()}, ignore_index=True)
            #
            # df.to_csv('images_editing_features.csv')

            images, img_path_image = batch
            images = images.to(device)
            if flags.debug:
                T.ToPILImage()(images[0, ...]).show()
            # ===================forward=====================
            outputs = model(images)
            input_contrast = batch_get_contrast(images)
            output_contrast = batch_get_contrast(outputs)
            input_mean_rgb = batch_get_mean_rgb(images)
            output_mean_rgb = batch_get_mean_rgb(outputs)
            input_contrast_temp = input_contrast.clone().detach().cpu().numpy()
            output_contrast_temp = output_contrast.clone().detach().cpu().numpy()
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
            global_step += 1
            writer.add_scalar('contrast_loss', contrast_loss, global_step)
            writer.add_scalar('mean_rgb_loss', mean_rgb_loss, global_step)
            writer.add_scalar('reconstruction_loss', reconstruction_loss, global_step)
            writer.add_scalar('loss', loss, global_step)
            train_loss += loss.item() * images.size(0)
            if batch_idx % 50 == 0:
                print('Epoch: {} \tBatch: {} \tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    loss.item()
                ))
                # save model weights
                if batch_idx % 100 == 0:
                    print('Saving model weights')
                    Path(f'./model_weights/{flags.timestamp}').mkdir(parents=True, exist_ok=True)
                    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict()},
                               f'./model_weights/{flags.timestamp}/conv_autoencoder_epoch_{epoch}_step_{global_step}_{flags.timestamp}.pt')

            # df.iloc[len(df):len(df)+len(img_path_image),:] = [img_path_image, input_contrast, input_mean_rgb[:, 0], input_mean_rgb[:, 1], input_mean_rgb[:, 2]]
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
