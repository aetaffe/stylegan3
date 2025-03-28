# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, cutmix=False):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.cutmix             = cutmix
        self.steps              = 0
        self.huber_loss         = torch.nn.HuberLoss()

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        augImg = img.detach()
        logits, dec_logits = self.D(img, c, update_emas=update_emas)
        return logits, dec_logits, augImg

    def _cutmix_coordinates(self, height, width, alpha = 1.):
        # Not sure about this...need to look into it.
        # I got it from: https://github.com/lucidrains/unet-stylegan2
        lam = np.random.beta(alpha, alpha)

        cx = np.random.uniform(0, width)
        cy = np.random.uniform(0, height)
        w = width * np.sqrt(1 - lam)
        h = height * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, width)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, height)))

        return ((y0, y1), (x0, x1)), lam

    def _cutmix(self, source, target, coors):
        source, target = map(torch.clone, (source, target))
        ((y0, y1), (x0, x1)), _ = coors
        source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
        return source

    def _mask_src_tgt(self, source, target, mask):
        return source * mask + (1 - mask) * target

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits, gen_dec_logits, _ = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain_dec = torch.nn.functional.softplus(-gen_dec_logits).mean(dim=(2,3)) # mean(-log(sigmoid(gen_logits)))
                loss_Gmain_enc = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain_enc + loss_Gmain_dec)
                loss_Gmain = loss_Gmain_dec + loss_Gmain_enc
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:

            # Dmain: Minimize logits for generated images.
            loss_Dgen = 0
            gen_aug_images = None
            gen_dec_logits_copy = None
            # if phase in ['Dmain', 'Dboth']:
            #     with torch.autograd.profiler.record_function('Dgen_forward'):
            #         gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
            #         gen_logits, gen_dec_logits, gen_aug_images = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma,
            #                                                                 update_emas=True)
            #         gen_dec_logits_copy = gen_dec_logits.detach()
            #         training_stats.report('Loss/scores/fake_enc', gen_logits)
            #         training_stats.report('Loss/signs/fake', gen_logits.sign())
            #         loss_Dgen_dec = torch.nn.functional.softplus(gen_dec_logits).mean(dim=(2, 3))
            #         training_stats.report('Loss/scores/fake_dec', loss_Dgen_dec)
            #         loss_Dgen_enc = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            #         loss_Dgen = loss_Dgen_enc + loss_Dgen_enc
            #     with torch.autograd.profiler.record_function('Dgen_backward'):
            #         loss_Dgen.mean().mul(gain).backward()


            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits, real_dec_logits, real_aug_images = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                cr_loss = 0
                if phase in ['Dmain', 'Dboth']:
                    with torch.autograd.profiler.record_function('Dgen_forward'):
                        gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                        gen_logits, gen_dec_logits, gen_aug_images = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma,
                                                                                update_emas=True)
                        gen_dec_logits_copy = gen_dec_logits.detach()
                        training_stats.report('Loss/scores/fake_enc', gen_logits)
                        training_stats.report('Loss/signs/fake', gen_logits.sign())
                        loss_Dgen_dec = torch.nn.functional.softplus(gen_dec_logits).mean(dim=(2, 3))
                        training_stats.report('Loss/scores/fake_dec', loss_Dgen_dec)
                        loss_Dgen_enc = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
                        loss_Dgen = loss_Dgen_enc + loss_Dgen_enc
                    # with torch.autograd.profiler.record_function('Dgen_backward'):
                    #     loss_Dgen.mean().mul(gain).backward()

                    loss_Dreal_enc = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_dec = torch.nn.functional.softplus(-real_dec_logits).mean(dim=(2,3))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal_enc)
                    loss_Dreal = loss_Dreal_enc + loss_Dreal_dec

                    dec_loss_coef = warmup(0, 1, 10000000, self.steps)
                    cutmix_prob = warmup(0, 0.25, 10000000, self.steps)
                    if self.cutmix and torch.rand(1).item() < cutmix_prob * self.augment_pipe.p:
                        image_size = real_img.shape[2]
                        mask = self._cutmix(
                            torch.ones_like(real_dec_logits),
                            torch.zeros_like(real_dec_logits),
                            self._cutmix_coordinates(image_size, image_size))

                        if torch.rand(1).item() > 0.5:
                            mask = 1 - mask

                        # mix(x, G(z), Mask)
                        cutmix_images = self._mask_src_tgt(real_aug_images, gen_aug_images, mask)
                        cutmix_logits, cutmix_dec_logits, _ = self.run_D(cutmix_images, gen_c)
                        loss_cutmix_enc = torch.nn.functional.softplus(cutmix_logits)
                        loss_cutmix_dec = torch.nn.functional.relu(1 + (mask * 2 - 1) * cutmix_dec_logits).mean(dim=(2,3))
                        loss_cutmix = loss_cutmix_enc + loss_cutmix_dec
                        training_stats.report('Loss/scores/cutmix', loss_cutmix)
                        training_stats.report('Loss/scores/cutmix_avg', loss_cutmix.mean().item())

                        # Consistency regularization
                        # mix(D_dec(x), D_dec(G(x)), Mask)
                        cr_cutmix_dec = self._mask_src_tgt(real_dec_logits, gen_dec_logits_copy, mask)
                        # cr_loss = torch.nn.functional.mse_loss(cutmix_dec_logits, cr_cutmix_dec) * 0.2
                        cr_loss = self.huber_loss(cutmix_dec_logits, cr_cutmix_dec)
                        training_stats.report('Loss/consistency_reg/loss', cr_loss)
                        training_stats.report('Loss/decoder_coefficient', dec_loss_coef)
                        loss_Dreal = loss_Dreal + (loss_cutmix * dec_loss_coef) + cr_loss * dec_loss_coef


                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1 + loss_Dgen).mean().mul(gain).backward()

#----------------------------------------------------------------------------
