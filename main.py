from pathlib import Path

import numpy as np
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple
from personal_utils.file_utils import append2file_name
import PIL
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg
from torchvision import transforms
from torch import nn
from torch.optim import Adam, LBFGS
from personal_utils.flags import flags
from torchvision.models import VGG


def preprocess_image(filename: str):



    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.CenterCrop(224),IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=NORM_MEANS, std=NORM_STD),

    ])
    proc_img = (preprocess(input_image))
    if flags.debug:
        plt.imshow(transforms.ToPILImage()(unnormalize(proc_img) / 255))
        plt.show()
    return proc_img


def unnormalize(img):
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    mean =NORM_MEANS
    std =NORM_STD
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    # inv_normalize = transforms.Normalize(
    # mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    # std=[1/0.229, 1/0.224, 1/0.255]
    # )
    #     img = inv_normalize(img)
    return img


class NN(VGG):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg.vgg19(pretrained=True).features[:29]
        self.style_layers = [0, 5, 10, 19, 28]
        # self.style_layers = [5, 10, 19, 28, 36]
        self.content_layers = [0]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x) -> Tuple[List[torch.Tensor]]:
        feat = []
        content = None
        for count, layer in enumerate(self.model):
            if count == 36:  # skip classifier
                break
            x = layer(x)
            if count in self.style_layers:
                feat.append(x)  # requires_grad_(False)
            if count in self.content_layers:
                content = x
        return feat, content


def setup_initial_image(content_image=None, init='content', requires_grad=False):
    if init == 'content':
        return preprocess_image(content_image,)


def segment_image():
    pass


def get_gram_mat(style_layers):
    gram_mat_list = []
    for i in style_layers:
        channels, width, height = i.shape
        i = i.squeeze().view(channels, width * height)
        gram_mat = i.matmul(i.T)
        gram_mat = gram_mat / (channels * width * height)
        gram_mat_list.append(gram_mat)
    if flags.debug:
        plt.imshow(transforms.ToPILImage()(gram_mat_list[-3]))
        plt.show()

    return gram_mat_list


def show_content_layers():
    pass


def compute_total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)



if __name__ == '__main__':
    NORM_MEANS = [123.675, 116.28, 103.53]
    NORM_STD = [1,1,1]

    wedding_content = 'data/wedding_content.jpg'
    wedding_style = 'data/wedding_style.jpg'
    wedding_style = '/home/bar/projects/personal/pytorch-neural-style-transfer-master/data/style-images/candy.jpg'
    device = 'cuda'
    writer = SummaryWriter('res')
    img_name = 'temp_img.pt'
    net = NN()
    initial_image = setup_initial_image(wedding_content)
    # initial_image = torch.ones((3, 224, 224), device="cuda")/10
    # noise_image.requires_grad_(True).cuda()
    initial_image = initial_image.cuda()
    initial_image.requires_grad_(True)
    net.eval().cuda()
    # initial_image = torch.randn(3,224,224,requires_grad=False,device=device)
    # initial_image=initial_image.to(device)
    # initial_image.requires_grad_(True)
    content_image = preprocess_image(wedding_content)

    n_channels, hight, width = content_image.shape

    image_size = (224, 224)
    _, target_content = net(content_image.cuda())
    target_style_list, _ = net(preprocess_image(wedding_style).cuda())
    del _
    target_gram_mat_list = get_gram_mat(target_style_list)

    if Path(img_name).exists() and False :
        initial_image = torch.load(img_name)
    opt = Adam([initial_image], lr=10)

    steps = 2000
    for count in tqdm(range(steps), total=steps):
        style_loss = .0
        style_list, content = net(initial_image)
        n_channels, hight, width = initial_image.shape
        content_loss = torch.nn.MSELoss(reduction='mean')(content, target_content)
        # torch.mean((content - target_content_list) ** 2) / (n_channels * hight * width)
        img_gram_mat_list = get_gram_mat(style_list)
        for curr_style, target_style in zip(img_gram_mat_list, target_gram_mat_list):
            style_loss = torch.nn.MSELoss(reduction='sum')(curr_style,target_style)
        var_loss = compute_total_variation_loss(initial_image.unsqueeze(0), 1)
        loss = style_loss/1e5+ 1e3 * content_loss + var_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if count % 50 == 0:
            writer.add_scalar('Loss', loss, count)
            print(loss)
            # im: Image.Image = transforms.ToPILImage()(initial_image)
            # im.save(f'res/images/image_{count}.jpg')
            # # res= seaborn.histplot(np.array(im)[:,:,0])
            # writer.add_image('img', initial_image)
            # writer.add_histogram('asd', np.array(im)[:, :, 0], count)
            img = transforms.ToPILImage()(unnormalize(initial_image.clone().detach() )/ 255)
            plt.imshow(img)
            plt.show()
            torch.save(initial_image, img_name)
    writer.close()
