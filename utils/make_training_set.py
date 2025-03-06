import os
import glob
import shutil
from pathlib import Path


def copy_training_data(img_file, dest, ctr):
    shutil.copy2(img_file, f'{dest}/images/train')
    shutil.move(f'{dest}/images/train/{img_file.split("/")[-1]}', f'{dest}/images/train/{ctr}.png')
    shutil.copy2(img_file.replace('.png', '.txt'), f'{dest}/labels/train')
    shutil.move(f'{dest}/labels/train/{img_file.split("/")[-1].replace(".png", ".txt")}',
                f'{dest}/labels/train/{ctr}.txt')

def copy_validation_data(img_file, dest, ctr):
    shutil.copy2(img_file, f'{dest}/images/val')
    shutil.move(f'{dest}/images/val/{img_file.split("/")[-1]}', f'{dest}/images/val/{ctr}.png')
    shutil.copy2(img_file.replace('.png', '.txt'), f'{dest}/labels/val')
    shutil.move(f'{dest}/labels/val/{img_file.split("/")[-1].replace(".png", ".txt")}',
                f'{dest}/labels/val/{ctr}.txt')


if __name__ == '__main__':
    train_split = 0.70
    val_split = 1 - train_split
    num_synthetic_train = 1000
    num_synthetic_val = round(num_synthetic_train * val_split)

    real_source = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-no-phantom-512x512_YOLO'
    synthetic_source = '/media/alex/1TBSSD/research_gans/Generated_Data/SyntheticDataset'
    dest = f'{real_source}/{real_source.split("/")[-1]}_train_val'
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

    for img_file in glob.glob(f'{real_source}/*.png'):
        if img_ctr < num_real:
            copy_training_data(img_file, dest, train_ctr)
            train_ctr += 1
        else:
            copy_validation_data(img_file, dest, val_ctr)
            val_ctr += 1
        img_ctr += 1

    img_ctr = 0
    for img_file in glob.glob(f'{synthetic_source}/*.png'):
        if img_ctr < num_synthetic_train:
            copy_training_data(img_file, dest, train_ctr)
            train_ctr += 1
        else:
            copy_validation_data(img_file, dest, val_ctr)
            val_ctr += 1
        img_ctr += 1









