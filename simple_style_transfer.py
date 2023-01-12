"""

a short script that uses vgg16 to apply image style transfer to a given image

"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from personal_utils.deep_learning_image_utils import (
    compute_total_variation_loss,
    get_gram_mat,
    unnormalize,
)
from personal_utils.flags import flags
from torch.optim import Adam
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
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STD),
        ]
    )
    proc_img = preprocess(input_image)
    return proc_img


class NN(VGG):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg.vgg19(pretrained=True).features[:29]
        self.style_layers = [0, 5, 10, 19, 28]
        # self.style_layers = [5, 10, 19, 28, 36]
        self.content_layers = [0]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x) -> Tuple[List[torch.Tensor], torch.Tensor]:
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


if __name__ == "__main__":
    NORM_MEANS = [123.675, 116.28, 103.53]
    NORM_STD = [1, 1, 1]

    content_img = "data/content_img.jpg"
    style_img = "data/style_img.jpg"
    device = "cuda"
    writer = SummaryWriter("res")
    net = NN()

    initial_image = preprocess_image(content_img)
    initial_image = initial_image.cuda()
    initial_image.requires_grad_(True)
    net.eval().cuda()
    content_image = preprocess_image(content_img)

    n_channels, height, width = content_image.shape

    image_size = (224, 224)
    _, target_content = net(content_image.cuda())
    target_style_list, _ = net(preprocess_image(style_img).cuda())
    del _
    target_gram_mat_list = get_gram_mat(target_style_list)

    opt = Adam([initial_image], lr=10)

    steps = 2000
    for count in tqdm(range(steps), total=steps):
        style_loss = 0.0
        style_list, content = net(initial_image)
        n_channels, height, width = initial_image.shape
        content_loss = torch.nn.MSELoss(reduction="mean")(content, target_content)
        img_gram_mat_list = get_gram_mat(style_list)
        for curr_style, target_style in zip(img_gram_mat_list, target_gram_mat_list):
            style_loss = torch.nn.MSELoss(reduction="sum")(curr_style, target_style)
        var_loss = compute_total_variation_loss(initial_image.unsqueeze(0), 1)
        loss = style_loss / 1e5 + 1e3 * content_loss + var_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if count % 50 == 0:
            writer.add_scalar("Loss", loss, count)
            print(loss)
            img = transforms.ToPILImage()(
                unnormalize(initial_image.clone().detach()) / 255
            )
            plt.imshow(img)
            plt.show()
    writer.close()
