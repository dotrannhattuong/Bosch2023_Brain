from glob import glob
import os

data_files = sorted(glob('segment/imgs/*'))
label_files = sorted(glob('segment/masks/*'))


anno = open("train.txt", 'w')
for img, label in zip(data_files, label_files):
    anno.writelines(f'{img},{label}\n')