## Stylegan 2/3 with Segmentation Masks
#### Hello! This is a fork of the original [stylegan3](https://github.com/NVlabs/stylegan3) repository that I have modified for my research at UC Davis.


I have made the following changes to the original repo:
1. I have modified stylegan 2/3 to generate a segmentation mask in addition to images using the `--seg_mask=1` option.
2. I have created a version of stylegan2 with a U-Net discriminator. This can be used by setting the cfg option, e.g. `--cfg=stylegan2-unet`
3. I have modified the dataset tool to resize images (see below).


To train the model:
### 1. Create a dataset zip file using the dataset tool
I modified the dataset tool to resize images. Images can be resized via the `--resize-size` command line option. For example
```bash
python utils/dataset_tool.py --source=<source dir> --dest=<dest dir> --resize-size=512
```
See the original repo for details on how to use the dataset tool or run `python utils/dataset_tool.py --help`

### 2. Train the model using the dataset
Example
```bash
python  train.py --outdir=<out dir> --cfg=stylegan2 --data=/paht/to/dataset/zip --gpus=1 --batch=16 --gamma=15 --mirror=1 --snap=10 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --seg_mask=1 --p=0.3
```

