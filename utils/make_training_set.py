import os
import glob
import shutil
from pathlib import Path


def copy_rename(img_file, dest, ctr, is_img=True):
    if is_img:
        shutil.copy2(img_file, f'{dest}/{ctr}.png')
    else:
        shutil.copy2(img_file, f'{dest}/{ctr}.txt')


if __name__ == '__main__':
    train_split = 0.70
    val_split = 1 - train_split
    num_synthetic_train = 2000
    num_synthetic_val = round(num_synthetic_train * val_split)

    real_source = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-no-phantom-512x512_YOLO'
    synthetic_source = '/media/alex/1TBSSD/research_gans/Generated_Data/SyntheticData2'
    dest = f'{"/".join(real_source.split("/")[:-1])}/{real_source.split("/")[-1]}_train_val'
    num_images = len(glob.glob(f'{real_source}/*.png'))
    num_real = round(num_images * train_split)

    Path(dest).mkdir(parents=True, exist_ok=True)
    Path(f'{dest}/images/train').mkdir(parents=True, exist_ok=True)
    Path(f'{dest}/labels/train').mkdir(parents=True, exist_ok=True)
    Path(f'{dest}/images/val').mkdir(parents=True, exist_ok=True)
    Path(f'{dest}/labels/val').mkdir(parents=True, exist_ok=True)

    img_ctr = 0
    train_ctr = 0
    val_ctr = 0
    img_files = sorted(glob.glob(f'{real_source}/*.png'))
    for img_file in img_files:
        if img_ctr < num_real:
            copy_rename(img_file, f'{dest}/images/train', train_ctr)
            copy_rename(img_file.replace('.png', '.txt'), f'{dest}/labels/train', train_ctr, False)
            train_ctr += 1
        else:
            copy_rename(img_file, f'{dest}/images/val', val_ctr)
            copy_rename(img_file.replace('.png', '.txt'), f'{dest}/labels/val', val_ctr, False)
            val_ctr += 1
        img_ctr += 1

    img_ctr = 0
    synthetic_files = sorted(glob.glob(f'{synthetic_source}/*.png'))
    for img_file in synthetic_files:
        if img_ctr < num_synthetic_train:
            copy_rename(img_file, f'{dest}/images/train', train_ctr)
            copy_rename(img_file.replace('.png', '.txt'), f'{dest}/labels/train', train_ctr, False)
            train_ctr += 1
        else:
            break
        img_ctr += 1









