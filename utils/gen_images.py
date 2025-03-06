# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from metrics import metric_main
import legacy
import supervision as sv
import cv2
import time

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def get_training_set_kwargs(train_data_path: str) -> dict:
    dataset_class = 'training.dataset.ImageSegmentationDataset'
    dataset_kwargs = dnnlib.EasyDict(class_name=dataset_class, path=train_data_path, use_labels=False, max_size=None, xflip=False)
    return dataset_kwargs

def calculate_fid(generator, train_data_path, score_threshold):
    training_set_kwargs = get_training_set_kwargs(train_data_path)
    results_dict = metric_main.calc_metric(metric='fid50k_full_threshold', G=generator, dataset_kwargs=training_set_kwargs,
                                           device=torch.device('cuda'), score_threshold=score_threshold)
#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def draw_polygon(polygon, img):
    for poly in polygon:
        poly = np.array(poly)
        poly = poly.reshape((-1, 1, 2))
        cv2.polylines(img, [np.int32(poly)], True, (0, 0, 255), 1)
    return img
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--num', type=int, help='Number of images to generate.', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--label-dir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--image-dir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--score-threshold', help='Base threshold for the discriminator score', type=int, required=False)
def generate_images(
    network_pkl: str,
    # seeds: List[int],
    num: int,
    truncation_psi: float,
    noise_mode: str,
    label_dir: str,
    image_dir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    score_threshold: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        models = legacy.load_network_pkl(f)
        G = models['G_ema'].to(device=device)
        D = models['D_ema'].to(device=device)
        # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    total_imgs = 0
    skipped_imgs = 0
    # Generate images.
    num_generated_imgs = 0
    start_seed = 0
    seed_idx = start_seed
    score_threshold = score_threshold if score_threshold is not None else 0
    start_time = time.time()
    while num_generated_imgs < num:
        total_imgs += 1

        # z = torch.randn([1, G.z_dim]).to(device)
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # image shape 1 x 4 x 256 x 256
        z = torch.from_numpy(np.random.RandomState(seed_idx).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        score = torch.sigmoid(D(img=img, c=None)).cpu().numpy()[0][0] * 1000
        # print(f'seed: {seed_idx}, score: {score}')

        if score >= score_threshold:
            # image shape 1 x 256 x 256 x 4
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            rgb_img = img[0, :, :, :3]
            mask = np.where(img[0, :, :, 3].cpu().numpy() > 100, 255, 0)
            polygon = sv.mask_to_polygons(mask)
            if len(polygon) == 0:
                seed_idx += 1
                continue
            print(f'Generated image {num_generated_imgs}/{num} (score={score:.2f}, seed={seed_idx})')
            # mask = mask[:,:, None]
            # mask = mask.repeat(3, 2)
            rgb_img = rgb_img.cpu().numpy()
            # color_copy = np.zeros_like(rgb_img)
            # color_copy[:] = [255, 255, 0]
            # rgb_img = np.where(mask == 255, color_copy, rgb_img)
            # rgb_img = draw_polygon(polygon, rgb_img.copy())
            with open(f'{label_dir}/seed{num_generated_imgs}.txt', 'w') as f:
                for idx, p in enumerate(polygon):
                    f.write(f'{0}')
                    for point in p:
                        f.write(f' {point[0] / 512} {point[1] / 512}')
            (PIL.Image.fromarray(rgb_img, 'RGB')
             .save(f'{image_dir}/seed{num_generated_imgs}.png'))
            num_generated_imgs += 1
        else:
            # print('skipping seed', seed_idx)
            skipped_imgs += 1
        seed_idx += 1
    end_time = time.time()
    print(f'total images: {total_imgs}, skipped images: {skipped_imgs}, kept images: {total_imgs - skipped_imgs}, percent kept: {100*(1 - skipped_imgs/total_imgs):.2f}%')
    print(f'Time elapsed: {end_time - start_time:.2f} seconds')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
