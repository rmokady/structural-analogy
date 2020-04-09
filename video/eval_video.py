import models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data as Data
import math

import sys
from PIL import Image
import torchvision
import argparse
import random
from utils import adjust_scales2image, generate_noise2, load_trained_pyramid_mix
from imresize import imresize2
import os.path as osp
import torchvision.utils as vutils
from torch.utils.data import DataLoader


def draw_concat(Gs, reals, NoiseAmp, in_s, mode, fixed_noise, opt):
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            for G, real_curr, real_next, noise_amp in zip(Gs, reals, reals[1:], NoiseAmp):
                if not opt.add_inject or count >= opt.inject_level:
                    if count == 0:
                        G_z = in_s
                    z = fixed_noise[count]

                    if count == opt.inject_level and opt.add_inject:
                        z_in = noise_amp * z + real_curr.cuda()
                        G_z = real_curr.cuda()
                    else:
                        G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                        z_in = noise_amp * z + G_z
                    if count > opt.switch_scale:
                        G_z = G(z_in.detach())
                    else:
                        G_z = G(z_in.detach(), G_z)
                    G_z = imresize2(G_z.detach(), 1 / opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1

    return G_z


def init_models(opt):

    #generator initialization:
    netG = models.Generator_no_res(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    return netD, netG


def init_models_res(opt):
    # generator initialization:
    netG = models.Generator(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    return netD, netG


class Video_dataset(Data.Dataset):
    """Faces."""

    def __init__(self, root_dir, size, ext, opt):
        self.root_dir = root_dir
        self.size = size
        self.ext = ext

        self.data = {}

        for ii in range(0, opt.stop_scale + 1, 1):
            scale = math.pow(opt.scale_factor, opt.stop_scale - ii)

            s_size = math.ceil(scale * opt.img_size)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((s_size, s_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            for j in range(self.size):
                if j not in self.data:
                    self.data[j] = []

                img_name = os.path.join(self.root_dir, str(j) + self.ext)
                image = Image.open(img_name).convert('RGB')
                sample = transform(image)
                self.data[j].append(sample)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return tuple(self.data[idx]), idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id, if the value is -1, the cpu is used')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)

    # networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)  # pow(0.5,1/6))
    parser.add_argument('--noise_amp_a', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--noise_amp_b', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)

    # main arguments
    parser.add_argument('--video_dir', help='input image path', required=True)
    # parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--vid_ext', default='.jpg', help='ext for video frames')
    parser.add_argument('--switch_res', type=int, default=2)
    parser.add_argument('--inject_level', type=int, default=9)
    parser.add_argument('--add_inject', type=bool, default=True)
    parser.add_argument('--a2b', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--img_size', type=int, default=220)
    parser.add_argument('--load', default='./load/')
    parser.add_argument('--out', default='./out/')
    opt = parser.parse_args()

    if not os.path.exists(opt.out):
        os.makedirs(opt.out)

    torch.cuda.set_device(opt.gpu_id)

    opt.device = "cuda:%s" % opt.gpu_id
    opt.noise_amp_init = opt.noise_amp_a
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor

    adjust_scales2image(opt.img_size, opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.gpu_id == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    Gs_a = []
    reals_a = []
    NoiseAmp_a = []

    Gs_b = []
    reals_b = []
    NoiseAmp_b = []

    nfc_prev = 0
    scale_num = 0

    r_loss = nn.MSELoss()

    dataset_a = Video_dataset(opt.video_dir, opt.num_images, opt.vid_ext, opt)
    data_loader_a = DataLoader(dataset_a, shuffle=True, batch_size=1)

    fixed_noise = []
    size_arr = []
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)
        size_arr.append(math.ceil(scale * opt.img_size))
        fixed_noise.append(generate_noise2([1, 3, size_arr[-1], size_arr[-1]], device=opt.device))

    # print(size_arr)

    opt.switch_scale = opt.stop_scale - opt.switch_res

    opt.nzx = size_arr[0]
    opt.nzy = size_arr[0]
    in_s = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)

    scale_num = opt.stop_scale

    if opt.a2b:
        Gs_a, reals_a, NoiseAmp_a, Gs_b, reals_b, NoiseAmp_b = load_trained_pyramid_mix(opt)
    else:
        Gs_b, reals_b, NoiseAmp_b , Gs_b, reals_b, NoiseAmp_b = load_trained_pyramid_mix(opt)

    G_a = Gs_a[-1]
    G_b = Gs_b[-1]
    opt.noise_amp_a = NoiseAmp_a[-1]
    opt.noise_amp_b = NoiseAmp_b[-1]

    Gs_a = Gs_a[:len(Gs_a) - 1]
    Gs_b = Gs_b[:len(Gs_b) - 1]

    NoiseAmp_a = NoiseAmp_a[:len(NoiseAmp_a) - 1]
    NoiseAmp_b = NoiseAmp_b[:len(NoiseAmp_b) - 1]

    print("debug")
    print(len(Gs_a))
    print(len(size_arr))
    opt.nzx = size_arr[len(Gs_a)]
    opt.nzy = size_arr[len(Gs_a)]

    for data in data_loader_a:

        data_a, idx = data

        real_a = data_a[len(Gs_a)].cuda()

        # Create fake from noise
        noise_ = fixed_noise[len(Gs_a)]

        prev_a = draw_concat(Gs_a, list(data_a), NoiseAmp_a, in_s, 'rand', fixed_noise, opt)
        noise_a = opt.noise_amp_a * noise_ + prev_a

        if scale_num > opt.switch_scale:
            fake_a = G_a(noise_a.detach())
        else:
            fake_a = G_a(noise_a.detach(), prev_a.detach())

        if scale_num > opt.switch_scale:
            mix_g_b = G_b(fake_a)
        else:
            mix_g_b = G_b(fake_a, fake_a)

        vutils.save_image(mix_g_b.clone(), osp.join(opt.out, str(idx.item()) + ".png"), normalize=True)
        print(idx.item())
        if opt.debug:
            vutils.save_image(fake_a.clone(), osp.join(opt.out, str(idx.item()) + "_fake_a.png"), normalize=True)

        print("debug imgs saved, scale_num=%0d" % scale_num)
        sys.stdout.flush()











