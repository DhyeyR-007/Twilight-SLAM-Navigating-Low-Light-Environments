from enlighten_inference import EnlightenOnnxModel
import cv2
import os
import glob
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Get Options")
    parser.add_argument('--img_dir', type=str, help='Path to Image Directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Path to Save Enlightened Images')
    parser.add_argument('--img_type', type=str, default='png', help='Image file type')
    args = parser.parse_args()
    return args


class enlighten_module(object):
    def __init__(self, img_dir: str, save_dir: str, img_type: str = "png") -> None:
        self.model = EnlightenOnnxModel()
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.img_type = img_type

    def evaluate(self, img_file : str = None) -> None:
        if img_file is not None:
            img_path = os.path.join(self.img_dir, img_file)
            image = cv2.imread(img_path)
            enlightened_img = self.model.predict(image)
            cv2.imwrite(os.path.join(self.save_dir, img_file), enlightened_img)
        else:
            for file in tqdm(glob.glob(self.img_dir+"/*."+self.img_type)):
                image = cv2.imread(file)
                enlightened_img = self.model.predict(image)
                basename = os.path.basename(file)
                cv2.imwrite(os.path.join(self.save_dir, basename), enlightened_img)


if __name__ == '__main__':
    opt = get_args()
    enGAN = enlighten_module(img_dir=opt.img_dir, save_dir=opt.save_dir, img_type=opt.img_type)
    enGAN.evaluate()
