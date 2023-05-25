# import argparse
import os

import kornia
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import cv2
import argparse
from torch.utils.data import DataLoader

import models
from datasets import LowLightDatasetNoGT
from tools import saver


def get_args():
    parser = argparse.ArgumentParser(description="Get Options")
    parser.add_argument('--img_dir', type=str, help='Path to Image Directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Path to Save Enlightened Images')
    parser.add_argument('--img_type', type=str, default='png', help='Image file type')

    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='Num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')

    parser.add_argument('-m1', '--model1', type=str, default='IAN', help='Model1 Name')
    parser.add_argument('-m2', '--model2', type=str, default='ANSN', help='Model2 Name')
    parser.add_argument('-m3', '--model3', type=str, default='FuseNet', help='Model3 Name')
    parser.add_argument('-m4', '--model4', type=str, default='FuseNet', help='Model4 Name')

    parser.add_argument('-m1w', '--m1_weights', type=str, default="./checkpoints/IAN_335.pth", help='Path for model weight of IAN')
    parser.add_argument('-m2w', '--m2_weights', type=str, default="./checkpoints/ANSN_422.pth", help='Path for model weight of ANSN')
    parser.add_argument('-m3w', '--m3_weights', type=str, default="./checkpoints/FuseNet_MECAN_251.pth", help='Path for model weight of CAN')
    parser.add_argument('-m4w', '--m4_weights', type=str, default="./checkpoints/FuseNet_NFM_297.pth", help='Path for model weight of NFM')

    args = parser.parse_args()
    return args


class ModelBreadNet(nn.Module):
    def __init__(self, model1, model2, model3, model4, m1w, m2w, m3w, m4w):
        super().__init__()
        self.eps = 1e-6
        self.model_ianet = model1(in_channels=1, out_channels=1)
        self.model_nsnet = model2(in_channels=2, out_channels=1)
        self.model_canet = model3(in_channels=4, out_channels=2)
        self.model_fdnet = model4(in_channels=3, out_channels=1)
        self.load_weight(self.model_ianet, m1w)
        self.load_weight(self.model_nsnet, m2w)
        self.load_weight(self.model_canet, m3w)
        self.load_weight(self.model_fdnet, m4w)

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            ret = model.load_state_dict(state_dict, strict=True)
            # print(ret)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image):
        # Color space mapping
        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)

        # Illumination prediction
        texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
        texture_illumi = self.model_ianet(texture_in_down)
        texture_illumi = F.interpolate(texture_illumi, scale_factor=2, mode='bicubic', align_corners=True)

        # Illumination adjustment
        texture_illumi = torch.clamp(texture_illumi, 0., 1.)
        texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
        texture_ia = torch.clamp(texture_ia, 0., 1.)

        # Noise suppression and fusion
        texture_nss = []
        for strength in [0., 0.05, 0.1]:
            attention = self.noise_syn_exp(texture_illumi, strength=strength)
            texture_res = self.model_nsnet(torch.cat([texture_ia, attention], dim=1))
            texture_ns = texture_ia + texture_res
            texture_nss.append(texture_ns)
        texture_nss = torch.cat(texture_nss, dim=1).detach()
        texture_fd = self.model_fdnet(texture_nss)

        # Color adaption
        colors = self.model_canet(torch.cat([texture_in, cb_in, cr_in, texture_fd], dim=1))

        cb_out, cr_out = torch.split(colors, 1, dim=1)
        cb_out = torch.clamp(cb_out, 0, 1)
        cr_out = torch.clamp(cr_out, 0, 1)

        # Color space mapping
        image_out = kornia.color.ycbcr_to_rgb(torch.cat([texture_fd, cb_out, cr_out], dim=1))
        image_out = torch.clamp(image_out, 0, 1)

        return image_out


class bread_module(object):
    def __init__(self,
                 img_dir:str,
                 save_dir:str,
                 img_type:str = "png",
                 n_gpus:int = 1,
                 n_workers:int = 12,
                 batch_size:int = 1,
                 m1_name:str = "IAN",
                 m2_name:str = "ANSN",
                 m3_name:str = "FuseNet",
                 m4_name:str = "FuseNet",
                 m1_weights:str = "./checkpoints/IAN_335.pth",
                 m2_weights:str = "./checkpoints/ANSN_422.pth",
                 m3_weights:str = "./checkpoints/FuseNet_MECAN_251.pth",
                 m4_weights:str = "./checkpoints/FuseNet_NFM_297.pth"):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.img_type = img_type
        self.n_gpus = n_gpus
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.m1 = m1_name
        self.m2 = m2_name
        self.m3 = m3_name
        self.m4 = m4_name
        self.m1w = m1_weights
        self.m2w = m2_weights
        self.m3w = m3_weights
        self.m4w = m4_weights

        model1 = getattr(models, self.m1)
        model2 = getattr(models, self.m2)
        model3 = getattr(models, self.m3)
        model4 = getattr(models, self.m4)
        self.model = ModelBreadNet(model1, model2, model3, model4, self.m1w, self.m2w, self.m3w, self.m4w)

        if self.n_gpus > 0:
            self.model = self.model.cuda()
            if self.n_gpus > 1:
                self.model = nn.DataParallel(self.model)

    def evaluate(self, img_file : str = None) -> None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        else:
            torch.manual_seed(42)

        self.model.eval()

        if img_file is not None:
            img_path = os.path.join(self.img_dir, img_file)
            image = cv2.imread(img_path)
            enlightened_img = self.model.predict(image)
            cv2.imwrite(os.path.join(self.save_dir, img_file), enlightened_img)
        else:
            val_params = {'batch_size': self.batch_size,
                          'shuffle': False,
                          'drop_last': False,
                          'num_workers': self.n_workers}
            val_set = LowLightDatasetNoGT(self.img_dir)
            val_generator = DataLoader(val_set, **val_params)
            val_generator = tqdm(val_generator)

            for iter, (data, name) in enumerate(val_generator):
                saver.base_url = self.save_dir
                with torch.no_grad():
                    if self.n_gpus == 1:
                        data = data.cuda()
                    image_out = self.model(data)
                    saver.save_image(image_out, name=os.path.basename(name[0][:-4]))


if __name__ == '__main__':
    opt = get_args()
    bread = bread_module(img_dir=opt.img_dir, save_dir=opt.save_dir, img_type=opt.img_type,
                         n_gpus=opt.num_gpus, n_workers=opt.num_workers, batch_size=opt.batch_size,
                         m1_name=opt.model1, m1_weights=opt.m1_weights,
                         m2_name=opt.model2, m2_weights=opt.m2_weights,
                         m3_name=opt.model3, m3_weights=opt.m3_weights,
                         m4_name=opt.model4, m4_weights=opt.m4_weights)
    bread.evaluate()
