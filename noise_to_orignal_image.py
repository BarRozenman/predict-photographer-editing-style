"""# this script will take noise and an original image and train a neuron network
 to transform the noise image to the original one via gradient descent
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from pathlib import Path
from personal_utils.flags import flags
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import VGG
from torchvision.models import vgg
from tqdm import tqdm


def preprocess_image(filename: str):
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.mul(255)),
        ]
    )
    proc_img = preprocess(input_image)
    return proc_img


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """

    w_variance = torch.sum(torch.pow(img[:, :, :-1] - img[:, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :-1, :] - img[:, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def get_gram_mat(style_layers):
    gram_mat_list = []
    for i in style_layers:
        channels, width, height = i.shape
        i = i.squeeze().view(channels, width * height)
        gram_mat = i.matmul(i.T)
        gram_mat_list.append(gram_mat)
    if flags.debug:
        plt.imshow(transforms.ToPILImage()(gram_mat_list[-3]))
        plt.show()
    return gram_mat_list


class Mod(VGG):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg.vgg19(pretrained=True).features[:29]
        self.chosen_features = [0, 5, 10, 19, 28]
        self.chosen_features = [0, 5, 10, 19, 28]
        self.style_layers = [5, 10, 19, 28, 36]
        self.content_layers = [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = []
        content = None
        for count, layer in enumerate(self.model):
            x = layer(x)
            if count in self.chosen_features:
                feat.append(x)
            elif count in self.content_layers:
                content = x
        return feat, content


if __name__ == "__main__":
    writer = SummaryWriter("res")
    wedding_content = "data/wedding_content.jpg"
    wedding_style = "data/wedding_style.jpg"

    device = "cuda"
    content_image = preprocess_image(wedding_content).to(device)
    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_STD_NEUTRAL = [1, 1, 1]

    model = Mod()
    # input_image_path = '/home/bar/projects/personal/pytorch/images/picasso.jpg'
    # load_image = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), lambda x: x * 255,
    #                                  transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_STD_NEUTRAL)])
    # noise_image = torch.rand((3,224,224),requires_grad=True)
    # img: torch.Tensor = load_image(Image.open(input_image_path))
    # img: torch.Tensor = preprocess_image(input_image_path)
    initial_image = setup_initial_image(wedding_style)

    if False:
        if True:
            initial_image = initial_image.clone().to("cuda")
            initial_image.add_(torch.rand((3, 224, 224), device="cuda") * 255)
            initial_image.requires_grad_(True).cuda()
        else:
            initial_image = torch.ones((3, 224, 224), device="cuda") / 10
            initial_image.requires_grad_(True).cuda()

    # fig, axs = plt.subplots(3,3)
    # fig.suptitle('Vertically stacked subplots')
    # loss_func = MSELoss()
    # for c,(i,j) in enumerate(itertools.product(range(3),range(3))):
    #     axs[i,j].imshow(y[c,:,:])
    im_name = "temp_img_.pt"

    if Path(im_name).exists() and False:
        initial_image = torch.load(im_name)

    # plt.show()
    opt = optim.Adam([initial_image], lr=1e1)
    # opt =optim.LBFGS([noise_image],lr=0.05)

    initial_image = initial_image.cuda()
    # noise_image = noise_image.cuda()
    model.eval().cuda()
    # x=model(img)
    # plt.imshow(transforms.ToPILImage()(x[9,:,:]))
    # plt.show()
    alpha = 1
    beta = 0.01
    running_loss = []
    device = "cuda"
    target_style, _ = model(initial_image)
    _, target_content = model(content_image)

    # noise_image.requires_grad_(True)
    # noise_image = noise_image.to(device=device)
    # noise_image.requires_grad_(True)

    # add loss for extreme color changes in pixels in the image with pytorch
    for i in tqdm(range(2000)):
        loss = 0
        target_style, _ = model(initial_image)

        curr_style, curr_content = model(initial_image)
        # y, target_content = model(content_image)
        target_gram_mat_list = get_gram_mat(curr_style)
        # curr_gram_mat_list = get_gram_mat(y)

        for orig_feat, noise_feat in zip(curr_style, target_style):
            n_channels, hight, width = orig_feat.shape
            # loss_1 = torch.mean((orig_feat - noise_feat) ** 2)
            # gram matrix:
            loss_1 = 0
            gram_1 = torch.matmul(
                orig_feat.view(n_channels, hight * width),
                orig_feat.view(n_channels, hight * width).T,
            )
            gram_2 = torch.matmul(
                noise_feat.view(n_channels, hight * width),
                noise_feat.view(n_channels, hight * width).T,
            )
            loss_2 = torch.mean((gram_1 - gram_2) ** 2) / (
                gram_1.shape[0] * gram_1.shape[1]
            )
            # loss_3 = 1/tv_loss(noise_image,0.001)
            loss_3 = 0
            loss += alpha * loss_1 + beta * loss_2 + loss_3
            # running_loss.append(loss.detach().cpu())
        # if np.mean(running_loss[-5:]) <= loss+0.0001:
        #     torch.save(noise_image, im_name)
        #     break

        # loss= loss_func(x,y)
        # loss = torch.mean((x[9]-y[9])**2)
        opt.zero_grad()
        loss.backward()

        opt.step()
        if i % 50 == 0:
            writer.add_scalar("Loss", loss, i)
            print(loss)
            im: Image.Image = transforms.ToPILImage()(initial_image)
            im.save(f"res/images/image_{i}.jpg")
            # res= seaborn.histplot(np.array(im)[:,:,0])
            writer.add_image("img", initial_image)
            writer.add_histogram("asd", np.array(im)[:, :, 0], i)

    torch.save(initial_image, im_name)

    writer.close()

    a = 1
