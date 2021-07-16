import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle
import random

import rfinder.data.converter as dc

"""
Easily run with:
quick-sbatch -t 4:0 -s 'source ~/Documents/dotfiles/ComputeCanada/env-setups/rfi.sh && source ~/envs/rfi/bin/activate && python ~/RFI/spawc21-bounding-boxes/make_test_set.py'
"""

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def line_count(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )





def main(args):
    load_dotenv()  # take environment variables from .env.

    input_dir = os.getenv('TRAIN_DIR')+'/'
    extension = '.sigmf-meta'
    items = list(Path(input_dir).glob('*'+extension))

    # Only take 10%
    items = random.choices(items, k=len(items)//10)


    print(f"PREPARING TO BUILD SPLIT APART SQUARE FILES")
    output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/'

    square_size=1024

    bounds_dict=defaultdict()
    with open(output_dir+"test.txt", "w") as test_list_file:

        for i, item in enumerate(items):
            print(f' {i+1}/{len(items)}')

            filename = str(item.resolve())
            
            imgs, labels_per_img, bounds_per_img = dc.sigmf_to_labelled_images(filename, img_w=square_size,
                                        img_h=square_size, NFFT=square_size, noverlap=square_size//2,
                                        img_overlap=square_size//2)

            imgs_maxs = [np.max(img) for img in imgs]
            vmax = max(imgs_maxs)
            imgs_mins = [np.min(img) for img in imgs]
            vmin = min(imgs_mins)

            imgs_labels_bounds_iter = zip(imgs, labels_per_img, bounds_per_img)
            for i in range(len(imgs)):
                img, labels, bound = next(imgs_labels_bounds_iter)

                # Skips empty images
                if args.skip_empty and len(labels)==0:
                    continue

                output_img_fname = item.stem+f'-{i}.png'
                plt.imsave(output_dir+'Images/'+output_img_fname, img, vmin=vmin, vmax=vmax)
                test_list_file.write(f"\n{output_dir+output_img_fname}")
                bounds_dict[output_img_fname] = bound        

                with open(output_dir+'Images/'+item.stem+f'-{i}.txt', 'a') as label_file:
                    for j, label in enumerate(labels):
                        labels[j] = label.scale(cy=1.0/square_size, cx=1.0/square_size, w=1.0/square_size, h=1.0/square_size)
                        assert np.max(label.as_list()) <= 1.0, f"Normalization did not work: {label.as_list()}\n  for file: {filename}"
                        label_file.write(f'0 {label.cx} {label.cy} {label.w} {label.h}\n')

    with open(output_dir+'bounds.pkl', 'wb') as f:
        pickle.dump(bounds_dict, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes test set out of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    # parser.add_argument('--continue', default=False, action='store_true',
    #     help='Whether to continue from where left off')
    parser.add_argument('--skip-empty', default=False, action='store_true',
        help='Whether to write images and annotation files that have nothing in them (noise)')
    args = parser.parse_args()
    main(args)