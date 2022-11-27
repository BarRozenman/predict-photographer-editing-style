from torch.nn import MSELoss
from tqdm import tqdm
from typing import List, Tuple

import PIL
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from personal_utils.flags import flags
from torchvision.models import VGG


def preprocess_image(filename: str):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.mul(255)),

    ])
    proc_img = (preprocess(input_image))
    # plt.imshow(transforms.ToPILImage()(unnormalize(proc_img/255)))
    # plt.show()
    return proc_img


def unnormalize(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
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
        self.chosen_features = [0, 5, 10, 19, 28]
        self.style_layers = [5, 10, 19, 28, 36]
        self.content_layers = [1]
        print(self.model)

    def forward(self, x) -> Tuple[List[torch.Tensor]]:
        feat = []
        content = None
        for count, layer in enumerate(self.model):
            if count == 36:  # skip classifier
                break
            x = layer(x)
            if count in self.style_layers:
                feat.append(x)#requires_grad_(False)
            elif count in self.content_layers:
                content = x
        return feat, content


def setup_initial_image(content_image=None, init='content', requires_grad=False):
    if init == 'content':
        b= preprocess_image(content_image).to(device)
        a=b.requires_grad_(True)
        # return a
        return preprocess_image(content_image)


def segment_image():
    pass


def get_gram_mat(style_layers):
    gram_mat_list = []
    for i in style_layers:
        channels, width, height = i.shape
        i = i.squeeze().view(channels, width * height)
        gram_mat = i.matmul(i.T)
        print(gram_mat.is_leaf)
        try:
            gram_mat_list.append(gram_mat.requires_grad_(False))
            q=1
        except:
            gram_mat_list.append(gram_mat)
    if flags.debug:
        plt.imshow(transforms.ToPILImage()(gram_mat_list[-3]))
        plt.show()
    return gram_mat_list


def show_content_layers():
    pass


if __name__ == '__main__':
    wedding_content = 'data/wedding_content.jpg'
    wedding_style = 'data/wedding_style.jpg'
    device = 'cuda'
    net = NN()
    # initial_image = setup_initial_image(wedding_content)
    initial_image = torch.ones((3, 224, 224), device="cuda")/10
    # noise_image.requires_grad_(True).cuda()
    initial_image = initial_image.cuda()
    net.eval().cuda()

    # initial_image = torch.randn(3,224,224,requires_grad=False,device=device)
    # initial_image=initial_image.to(device)
    # initial_image.requires_grad_(True)
    content_image = preprocess_image(wedding_content)
    content_image.requires_grad_(False)

    _, target_content_list = net(content_image.cuda())
    # target_content_list.requires_grad_(False)
    target_style_list, _ = net(preprocess_image(wedding_style).cuda())
    del _
    opt = Adam([initial_image], lr=0.001)
    target_gram_mat_list = get_gram_mat(target_style_list)
    n_channels, hight, width = content_image.shape

    # target_gram_mat_list= [x.requires_grad_(False) for x in target_gram_mat_list]
    image_size = (224, 224)
    net.requires_grad_(False)

    # loss_func =
    # mse_loss = MSELoss(reduction='mean')
    for count in tqdm(range(50)):
        style_loss = 0
        content_loss=0

        # with torch.no_grad():
        style_list, content = net(initial_image)
        # content.requires_grad_(False)
        # for curr_content,target_content in zip(curr_content_list,target_content_list):
        content_loss = torch.mean((content - target_content_list) ** 2) / len(content)
        # img_gram_mat_list = get_gram_mat(style_list)
        # for curr_style, target_style in zip(img_gram_mat_list, target_gram_mat_list):
        #     style_loss += 00000000.1 * torch.mean(((curr_style - target_style) ** 2) / (image_size[0] * image_size[1]))
        loss = style_loss + content_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
        if count % 3 == 0:
            img = transforms.ToPILImage()(unnormalize(initial_image.detach()/255))
            plt.imshow(img)
            plt.show()

# im=transforms.ToPILImage()(res[1][0,0,...])
# plt.imshow(im)
# plt.show()
