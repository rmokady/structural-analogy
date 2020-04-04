from imresize import imresize
import math
import torch
import torch.nn as nn
import os
import sys



def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def generate_noise2(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    noise = []
    for i in range(size[0]):
        noise.append(generate_noise(size[1:], num_samp=1, device='cuda', type='gaussian', scale=1).squeeze(0))

    res = torch.stack(noise, dim=0)

    return res


def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if(os.path.exists(opt.load)):
        Gs = torch.load('%s/Gs.pth' % opt.load, map_location=opt.device)
        Zs = torch.load('%s/Zs.pth' % opt.load)
        reals = torch.load('%s/reals.pth' % opt.load)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.load)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def load_trained_pyramid_mix(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if(os.path.exists(opt.load)):
        Gs_a = torch.load('%s/Gs_a.pth' % opt.load, map_location=opt.device)
        Zs_a = torch.load('%s/Zs_a.pth' % opt.load)
        reals_a = torch.load('%s/reals_a.pth' % opt.load)
        NoiseAmp_a = torch.load('%s/NoiseAmp_a.pth' % opt.load, map_location=opt.device)

        Gs_b = torch.load('%s/Gs_b.pth' % opt.load, map_location=opt.device)
        Zs_b = torch.load('%s/Zs_b.pth' % opt.load)
        reals_b = torch.load('%s/reals_b.pth' % opt.load)
        NoiseAmp_b = torch.load('%s/NoiseAmp_b.pth' % opt.load, map_location=opt.device)

    else:
        print('no appropriate trained model is exist, please train first')
        sys.exit()
    opt.mode = mode
    return Gs_a, Zs_a, reals_a, NoiseAmp_a, Gs_b, Zs_b, reals_b, NoiseAmp_b