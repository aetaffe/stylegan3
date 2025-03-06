import os
import glob
import json
from pathlib import Path
import shutil

if __name__ == '__main__':
    source = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-no-phantom-512x512'
    outdir = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-no-phantom-512x512_YOLO'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    file_count = 0
    files_without = 0
    folders = os.listdir(source)
    for folder in folders:
        json_files = glob.glob(source + '/' + folder + '/*.json')
        for json_file in json_files:
            img_file = json_file.split('/')[-1].replace('.json', '.png')
            print(img_file)
            with open(json_file, 'r') as f:
                annotation = json.load(f)
                polygons = []
                width = float(annotation['imageWidth'])
                height = float(annotation['imageHeight'])
                for shape in annotation['shapes']:
                    if shape['label'] == 'AimingBeam':
                        if len(shape['points']) > 2:
                            polygons.append(shape['points'])
                if len(polygons) > 0:
                    shutil.copy2(f'{source}/{folder}/{img_file}', f'{outdir}/{file_count}.png')
                    with open(f'{outdir}/{file_count}.txt', 'w') as outfile:
                        outfile.write(f'0')
                        for polygon in polygons:
                            for point in polygon:
                                outfile.write(f' {point[0] / width } {point[1] / height }')
                    file_count += 1
                else:
                    print(f'no annotation for {img_file}')
                    files_without += 1


    print(f'{files_without} files without annotation')
    print(f'{file_count} files created')



