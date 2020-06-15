
import argparse
from PIL import Image, ImageOps
import torchvision.utils as vutils
import torchvision
import torch
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--quant_level', type=int, default=2)
    parser.add_argument('--resize', type=int, default=220)

    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    files = [f for f in os.listdir(args.root_dir) if f.endswith(args.ext)]

    for f in files:
        img_name = os.path.join(args.root_dir, f)
        image = Image.open(img_name).convert('L').convert('RGB')
        image = ImageOps.invert(image)
        sample = transform(image)

        quant = torch.trunc((-0.01 + sample) * args.quant_level)
        quant = quant / args.quant_level

        out_name = os.path.join(args.out, f.replace(args.ext, "png"))
        print(out_name)
        vutils.save_image(quant, out_name, normalize=True)