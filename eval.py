
import models
import os
import torch.utils.data
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



def draw_concat(Gs, reals, NoiseAmp, in_s, mode, inject_level, opt):
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            for G,real_curr,real_next,noise_amp in zip(Gs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = generate_noise2([1, 3, real_curr.shape[2], real_curr.shape[3]], device=opt.device)
                    G_z = in_s
                else:
                    z = generate_noise2([1, opt.nc_z,real_curr.shape[2], real_curr.shape[3]], device=opt.device)

                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                if count == inject_level:
                    z_in = noise_amp*z + real_curr.cuda()
                else:
                    z_in = noise_amp*z+G_z
                if count > opt.switch_scale:
                    G_z = G(z_in.detach())
                else:
                    G_z = G(z_in.detach(), G_z)
                G_z = imresize2(G_z.detach(),1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
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

def transform_input(img_path, opt):

    res = []
    image = Image.open(img_path).convert('RGB')
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)

        s_size = math.ceil(scale * opt.img_size)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((s_size, s_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        sample = transform(image)
        res.append(sample.unsqueeze(0))

    return res


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
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=18)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)

    #main arguments
    parser.add_argument('--input_a', help='input image path', required=True)
    parser.add_argument('--input_b', help='input image path', required=True)
    parser.add_argument('--switch_res', type=int, default=2, help='how many levels will not be residual')
    parser.add_argument('--img_size', type=int, default=220, help='image size of the output')
    parser.add_argument('--out', required=True)
    parser.add_argument('--print_interval', type=int, default=1000)
    parser.add_argument('--inject_levels', type=int, default=5)
    parser.add_argument('--add_inject', type=bool, default=True)
    parser.add_argument('--load', required=True)
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

    data_a = transform_input(opt.input_a, opt)
    data_b = transform_input(opt.input_b, opt)

    size_arr = []
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)
        size_arr.append(math.ceil(scale * opt.img_size))

    opt.switch_scale = opt.stop_scale - opt.switch_res

    opt.nzx = size_arr[0]
    opt.nzy = size_arr[0]
    in_s = torch.full([1, opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)

    scale_num = opt.stop_scale

    Gs_a, reals_a, NoiseAmp_a, Gs_b, reals_b, NoiseAmp_b = load_trained_pyramid_mix(opt)

    G_a = Gs_a[-1]
    G_b = Gs_b[-1]
    opt.noise_amp_a = NoiseAmp_a[-1]
    opt.noise_amp_b = NoiseAmp_b[-1]

    Gs_a = Gs_a[:len(Gs_a)-1]
    Gs_b = Gs_b[:len(Gs_b)-1]

    NoiseAmp_a = NoiseAmp_a[:len(NoiseAmp_a)-1]
    NoiseAmp_b = NoiseAmp_b[:len(NoiseAmp_b)-1]

    scale_num = opt.stop_scale - opt.inject_levels + 1

    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(opt.stop_scale / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(opt.stop_scale / 4)), 128)

        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

        opt.nzx = size_arr[len(Gs_a)]
        opt.nzy = size_arr[len(Gs_a)]

        real_a = data_a[len(Gs_a)].cuda()
        real_b = data_b[len(Gs_b)].cuda()

        noise_ = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

        prev_a = draw_concat(Gs_a, list(data_a), NoiseAmp_a, in_s, 'rand', scale_num, opt)
        noise_a = opt.noise_amp_a * noise_ + prev_a

        noise_ = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

        prev_b = draw_concat(Gs_b, list(data_b), NoiseAmp_b, in_s, 'rand', scale_num, opt)
        noise_b = opt.noise_amp_b * noise_ + prev_b

        if opt.stop_scale > opt.switch_scale:
            fake_a = G_a(noise_a.detach())
            fake_b = G_b(noise_b.detach())
        else:
            fake_a = G_a(noise_a.detach(), prev_a.detach())
            fake_b = G_b(noise_b.detach(), prev_b.detach())

        if opt.stop_scale > opt.switch_scale:
            mix_g_a = G_a(fake_b)
            mix_g_b = G_b(fake_a)
        else:
            mix_g_a = G_a(fake_b, fake_b)
            mix_g_b = G_b(fake_a, fake_a)

        if opt.stop_scale > opt.switch_scale:
            eval_real_a = G_a(real_b)
            eval_real_b = G_b(real_a)
        else:
            eval_real_a = G_a(real_b, real_b)
            eval_real_b = G_b(real_a, real_a)

        vutils.save_image(fake_a.clone(), osp.join(opt.out, "fake_a_" + str(scale_num) +".png"), normalize=True)
        vutils.save_image(mix_g_a.clone(), osp.join(opt.out, "a2b_inject_" + str(scale_num) + ".png"),
                          normalize=True)
        vutils.save_image(fake_b.clone(), osp.join(opt.out, "fake_b_" + str(scale_num) + ".png"),
                          normalize=True)
        vutils.save_image(mix_g_b.clone(), osp.join(opt.out, "b2a_inject_" + str(scale_num) + ".png"),
                          normalize=True)
        if scale_num == opt.stop_scale:
            vutils.save_image(eval_real_a.clone(), osp.join(opt.out, "b2a_from_real_" + ".png"),
                              normalize=True)
            vutils.save_image(eval_real_b.clone(), osp.join(opt.out,"b2a_from_real_" + ".png"),
                              normalize=True)
            vutils.save_image(real_a.clone(), osp.join(opt.out, "real_a_" + ".png"),
                              normalize=True)
            vutils.save_image(real_b.clone(), osp.join(opt.out, "real_b_" + ".png"),
                              normalize=True)

        print("imgs saved, scale_num=%0d" % (scale_num))
        sys.stdout.flush()
        scale_num += 1











