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
from utils import adjust_scales2image, generate_noise2, calc_gradient_penalty
from imresize import imresize2
import os.path as osp
import torchvision.utils as vutils
from torch.utils.data import DataLoader


def draw_concat(Gs,reals, NoiseAmp, in_s, mode, opt):
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
                z_in = noise_amp*z+G_z
                if count > opt.switch_scale:
                    G_z = G(z_in.detach())
                else:
                    G_z = G(z_in.detach(), G_z)
                G_z = imresize2(G_z.detach(),1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1

        if mode == 'rec':
            count = 0
            for G,real_curr,real_next,noise_amp in zip(Gs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    size = list(real_curr.size())
                    #print(size)
                    G_z = generate_noise2(size, device=opt.device)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                if count > opt.switch_scale:
                    G_z = G(G_z)
                else:
                    G_z = G(G_z, G_z)
                G_z = imresize2(G_z.detach(), 1/opt.scale_factor,opt)
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
        return tuple(self.data[idx])


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
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=22)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=24000, help='number of epochs to train per scale')
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=1.0)
    parser.add_argument('--beta', type=float, help='cycle loss weight', default=1.0)
    parser.add_argument('--lambda_self', type=float, default=1.0)

    # main arguments
    parser.add_argument('--video_dir', help='input image path', required=True)
    parser.add_argument('--input_b', help='input image path', required=True)
    # parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--vid_ext', default='.jpg', help='ext for video frames')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--switch_res', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=220)
    parser.add_argument('--out', default='./out/')
    parser.add_argument('--print_interval', type=int, default=1000)

    opt = parser.parse_args()

    if not os.path.exists(opt.out):
        os.makedirs(opt.out)

    torch.cuda.set_device(opt.gpu_id)

    opt.device = "cuda:%s" % opt.gpu_id
    opt.niter_init = opt.niter
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

    opt.print_interval = int(opt.print_interval / opt.num_images)

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
    data_loader_a = DataLoader(dataset_a, shuffle=True, batch_size=opt.bs)

    data_b = transform_input(opt.input_b, opt)

    size_arr = []
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)
        size_arr.append(math.ceil(scale * opt.img_size))

    opt.switch_scale = opt.stop_scale - opt.switch_res

    opt.nzx = size_arr[0]
    opt.nzy = size_arr[0]
    in_s = torch.full([opt.bs, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)

    while scale_num < opt.stop_scale + 1:

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        if scale_num > opt.switch_scale:
            D_a, G_a = init_models(opt)
            D_b, G_b = init_models(opt)
            print("No Res !!!")
        else:
            D_a, G_a = init_models_res(opt)
            D_b, G_b = init_models_res(opt)
            print("Res !!!")

        if nfc_prev == opt.nfc:
            print("Load weights of last layer")
            G_a.load_state_dict(torch.load('%s/netG_a_%d.pth' % (opt.out, scale_num - 1)))
            D_a.load_state_dict(torch.load('%s/netD_a_%d.pth' % (opt.out, scale_num - 1)))
            G_b.load_state_dict(torch.load('%s/netG_b_%d.pth' % (opt.out, scale_num - 1)))
            D_b.load_state_dict(torch.load('%s/netD_b_%d.pth' % (opt.out, scale_num - 1)))

        optimizerD = optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(list(G_a.parameters()) + list(G_b.parameters()), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        n_iters = int(opt.niter / opt.num_images)

        opt.nzx = size_arr[len(Gs_a)]
        opt.nzy = size_arr[len(Gs_a)]

        noise_amount_a = 0
        noise_cnt_a = 0

        noise_amount_b = 0
        noise_cnt_b = 0

        i = 0

        for epoch in range(n_iters):
            for data_a in data_loader_a:

                real_a = data_a[len(Gs_a)].cuda()
                real_b = data_b[len(Gs_b)].cuda()

                # Create fake from noise
                noise_ = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

                if Gs_a == []:
                    noise_a = noise_
                    prev_a = torch.full([opt.bs, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                else:
                    prev_a = draw_concat(Gs_a, list(data_a), NoiseAmp_a, in_s, 'rand', opt)
                    noise_a = opt.noise_amp_a * noise_ + prev_a

                noise_ = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

                if Gs_b == []:
                    noise_b = noise_
                    prev_b = torch.full([opt.bs, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                else:
                    prev_b = draw_concat(Gs_b, list(data_b), NoiseAmp_b, in_s, 'rand', opt)
                    noise_b = opt.noise_amp_b * noise_ + prev_b

                if scale_num > opt.switch_scale:
                    fake_a = G_a(noise_a.detach())
                    fake_b = G_b(noise_b.detach())
                else:
                    fake_a = G_a(noise_a.detach(), prev_a.detach())
                    fake_b = G_b(noise_b.detach(), prev_b.detach())

                if Gs_a == []:
                    z_prev_a = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
                else:
                    z_prev_a = draw_concat(Gs_a, list(data_a), NoiseAmp_a, in_s, 'rec', opt)

                if epoch == 0 and i == 0:
                    if Gs_a == []:
                        opt.noise_amp_a = 1
                    else:
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real_a, z_prev_a))
                        opt.noise_amp_a = opt.noise_amp_init * RMSE / opt.bs

                if Gs_b == []:
                    z_prev_b = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
                else:
                    z_prev_b = draw_concat(Gs_b, list(data_b), NoiseAmp_b, in_s, 'rec', opt)

                if epoch == 0 and i == 0:
                    if Gs_b == []:
                        opt.noise_amp_b = 1
                    else:
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real_b, z_prev_b))
                        opt.noise_amp_b = opt.noise_amp_init * RMSE / opt.bs

                i += 1

                if scale_num > opt.switch_scale:
                    generated_a = G_a(z_prev_a.detach())
                    generated_b = G_b(z_prev_b.detach())
                else:
                    generated_a = G_a(z_prev_a.detach(), z_prev_a.detach())
                    generated_b = G_b(z_prev_b.detach(), z_prev_b.detach())

                if scale_num > opt.switch_scale:
                    mix_g_a = G_a(fake_b)
                    mix_g_b = G_b(fake_a)
                else:
                    mix_g_a = G_a(fake_b, fake_b)
                    mix_g_b = G_b(fake_a, fake_a)

                other_noise_a = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
                other_noise_b = generate_noise2([opt.bs, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

                noisy_real_b = opt.noise_amp_a * other_noise_a + real_b
                noisy_real_a = opt.noise_amp_b * other_noise_b + real_a

                if opt.lambda_self > 0.0:
                    if scale_num > opt.switch_scale:
                        self_a = G_a(noisy_real_b)
                        self_b = G_b(noisy_real_a)
                    else:
                        self_a = G_a(noisy_real_b, noisy_real_b)
                        self_b = G_b(noisy_real_a, noisy_real_a)

                #############################
                ####      Train D_a      ####
                #############################

                D_a.zero_grad()

                output = D_a(real_a).to(opt.device)
                errD_real = -1 * (2 + opt.lambda_self) * output.mean()  # -a
                errD_real.backward(retain_graph=True)

                output_a = D_a(mix_g_a.detach())
                output_a2 = D_a(fake_a.detach())
                if opt.lambda_self > 0.0:
                    output_a3 = D_a(self_a.detach())
                    output_a3 = output_a3.mean()
                else:
                    output_a3 = 0
                errD_fake_a = output_a.mean() + output_a2.mean() + opt.lambda_self * output_a3
                errD_fake_a.backward(retain_graph=True)

                gradient_penalty_a = calc_gradient_penalty(D_a, real_a, mix_g_a, opt.lambda_grad, opt.device)
                gradient_penalty_a += calc_gradient_penalty(D_a, real_a, fake_a, opt.lambda_grad, opt.device)
                if opt.lambda_self > 0.0:
                    gradient_penalty_a += opt.lambda_self * calc_gradient_penalty(D_a, real_a, self_a, opt.lambda_grad,
                                                                              opt.device)
                gradient_penalty_a.backward(retain_graph=True)


                #############################
                ####      Train D_b      ####
                #############################

                D_b.zero_grad()

                output = D_b(real_b).to(opt.device)
                errD_real = -1 * (2 + opt.lambda_self) * output.mean()  # -a
                errD_real.backward(retain_graph=True)

                output_b = D_b(mix_g_b.detach())
                output_b2 = D_b(fake_b.detach())
                if opt.lambda_self > 0.0:
                    output_b3 = D_b(self_b.detach())
                    output_b3 = output_b3.mean()
                else:
                    output_b3 = 0
                errD_fake_b = output_b.mean() + output_b2.mean() + opt.lambda_self * output_b3
                errD_fake_b.backward(retain_graph=True)

                gradient_penalty_b = calc_gradient_penalty(D_b, real_b, mix_g_b, opt.lambda_grad, opt.device)
                gradient_penalty_b += calc_gradient_penalty(D_b, real_b, fake_b, opt.lambda_grad, opt.device)
                if opt.lambda_self > 0.0:
                    gradient_penalty_b += opt.lambda_self * calc_gradient_penalty(D_b, real_b, self_b, opt.lambda_grad,
                                                                              opt.device)
                gradient_penalty_b.backward(retain_graph=True)

                optimizerD.step()

                #############################
                ####      Train G      ####
                #############################

                G_a.zero_grad()
                G_b.zero_grad()

                output_a = D_a(mix_g_a)
                output_a2 = D_a(fake_a)
                if opt.lambda_self > 0.0:
                    output_a3 = D_a(self_a)
                    output_a3 = output_a3.mean()
                else:
                    output_a3 = 0
                errG_a = -output_a.mean() - output_a2.mean() - opt.lambda_self
                errG_a.backward(retain_graph=True)

                output_b = D_b(mix_g_b)
                output_b2 = D_b(fake_b)
                if opt.lambda_self > 0.0:
                    output_b3 = D_b(self_a)
                    output_b3 = output_b3.mean()
                else:
                    output_b3 = 0
                errG_b = -output_b.mean() - output_b2.mean() - opt.lambda_self
                errG_b.backward(retain_graph=True)

                if opt.alpha > 0:
                    rec_loss_a = opt.alpha * r_loss(generated_a, real_a)
                    rec_loss_a.backward(retain_graph=True)

                    rec_loss_b = opt.alpha * r_loss(generated_b, real_b)
                    rec_loss_b.backward(retain_graph=True)

                if opt.beta > 0:
                    if scale_num > opt.switch_scale:
                        cycle_b = G_b(mix_g_a)
                        cycle_a = G_a(mix_g_b)
                    else:
                        cycle_b = G_b(mix_g_a, mix_g_a)
                        cycle_a = G_a(mix_g_b, mix_g_b)

                    cycle_loss_a = opt.beta * r_loss(cycle_a, fake_a)
                    cycle_loss_a.backward(retain_graph=True)

                    cycle_loss_b = opt.beta * r_loss(cycle_b, fake_b)
                    cycle_loss_b.backward(retain_graph=True)

                optimizerG.step()

            if (epoch+1) % opt.print_interval == 0:
                vutils.save_image(fake_a.clone(),
                                  osp.join(opt.out, str(scale_num) + "_fake_a_" + str(epoch) + ".png"),
                                  normalize=True)
                vutils.save_image(mix_g_a.clone(),
                                  osp.join(opt.out, str(scale_num) + "_b2a_" + str(epoch) + ".png"),
                                  normalize=True)

                if epoch == 0:
                    vutils.save_image(real_a.clone(),
                                      osp.join(opt.out, str(scale_num) + "_real_a_" + str(epoch) + ".png"),
                                      normalize=True)

                vutils.save_image(fake_b.clone(),
                                  osp.join(opt.out, str(scale_num) + "_fake_b_" + str(epoch) + ".png"),
                                  normalize=True)
                vutils.save_image(mix_g_b.clone(),
                                  osp.join(opt.out, str(scale_num) + "_a2b_" + str(epoch) + ".png"),
                                  normalize=True)
                if epoch == 0:
                    vutils.save_image(real_b.clone(),
                                      osp.join(opt.out, str(scale_num) + "_real_b_" + str(epoch) + ".png"),
                                      normalize=True)

                print("debug imgs saved, scale_num=%0d, epoch=%0d " % (scale_num, epoch))
                sys.stdout.flush()

        if scale_num == opt.stop_scale:
            vutils.save_image(fake_a.clone(), osp.join(opt.out, "final_fake_a_" + str(epoch) + ".png"),
                              normalize=True)
            vutils.save_image(mix_g_a.clone(), osp.join(opt.out, "final_b2a_" + str(epoch) + ".png"),
                              normalize=True)

            vutils.save_image(fake_b.clone(), osp.join(opt.out, "final_fake_b_" + str(epoch) + ".png"),
                              normalize=True)
            vutils.save_image(mix_g_b.clone(), osp.join(opt.out, "final_a2b_" + str(epoch) + ".png"),
                              normalize=True)


        Gs_a.append(G_a)
        NoiseAmp_a.append(opt.noise_amp_a)

        torch.save(Gs_a, '%s/Gs_a.pth' % (opt.out))
        torch.save(reals_a, '%s/reals_a.pth' % (opt.out))
        torch.save(NoiseAmp_a, '%s/NoiseAmp_a.pth' % (opt.out))

        torch.save(G_a.state_dict(), '%s/netG_a_%d.pth' % (opt.out, scale_num))
        torch.save(D_a.state_dict(), '%s/netD_a_%d.pth' % (opt.out, scale_num))

        Gs_b.append(G_b)
        NoiseAmp_b.append(opt.noise_amp_b)

        torch.save(Gs_b, '%s/Gs_b.pth' % (opt.out))
        torch.save(reals_b, '%s/reals_b.pth' % (opt.out))
        torch.save(NoiseAmp_b, '%s/NoiseAmp_b.pth' % (opt.out))

        torch.save(G_b.state_dict(), '%s/netG_b_%d.pth' % (opt.out, scale_num))
        torch.save(D_b.state_dict(), '%s/netD_b_%d.pth' % (opt.out, scale_num))

        print("Layer weights saved successfully")

        scale_num += 1
        nfc_prev = opt.nfc
        del D_a, G_a
        del D_b, G_b





