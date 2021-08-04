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
quick-sbatch -t 4:0 -s 'source ~/Documents/dotfiles/ComputeCanada/env-setups/rfi.sh && source ~/envs/rfi/bin/activate && python ~/RFI/spawc21-bounding-boxes/make_test_set.py - '
"""

def _make_gen(reader):
    """
    Turns a file reader into a generator. Utility function for `line_count()`.
    """
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def line_count(filename):
    """ Utility function counts the lines in a text file."""
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )



def main(args):

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        raise FileNotFoundError (f"Output directory {output_dir} does not exist")

    images_dir = output_dir / 'Images/'
    if not images_dir.exists():
        images_dir.mkdir()

    load_dotenv()  # take environment variables from .env.

    input_dir = os.getenv('TRAIN_DIR')+'/'
    extension = '.sigmf-meta'
    items = list(Path(input_dir).glob('*'+extension))

    # Randomly select items
    if args.percent<1:
        items = random.choices(items, k=int(len(items)*args.percent))
    if args.percent==0:
        raise ValueError ("Can't make a test set out of 0 percent of the training files")


    print(f"PREPARING TO BUILD SPLIT-APART SQUARE FILES")
    square_size=args.square_size

    bounds_dict=defaultdict()
    with open(output_dir / "test.txt", "w") as test_list_file:

        for i, item in enumerate(items):
            print(f' {i+1}/{len(items)}')

            filename = str(item.resolve())
            
            imgs, labels_per_img, bounds_per_img = dc.sigmf_to_windowed_images(filename, img_w=square_size,
                                        img_h=square_size, NFFT=square_size, noverlap=square_size//2,
                                        img_overlap=square_size//2,
                                        return_labels=True)

            imgs_maxs = [np.max(img) for img in imgs]
            vmax = max(imgs_maxs)
            imgs_mins = [np.min(img) for img in imgs]
            vmin = min(imgs_mins)

            imgs_labels_bounds_iter = zip(imgs, labels_per_img, bounds_per_img)
            for j in range(len(imgs)):
                img, labels, bound = next(imgs_labels_bounds_iter)

                # Skips empty images
                if args.skip_empty and len(labels)==0:
                    continue

                output_img_path = images_dir / (item.stem+f'-{j}.png')
                plt.imsave(output_img_path, img, vmin=vmin, vmax=vmax)
                bounds_dict[output_img_path.name] = bound        

                test_list_file.write(f'{output_img_path.resolve()}\n')

                output_txt_path = images_dir / (item.stem+f'-{j}.txt')
                with open(output_txt_path, 'a') as label_file:
                    for k, label in enumerate(labels):
                        labels[k] = label.scale(cy=1.0/square_size, cx=1.0/square_size, w=1.0/square_size, h=1.0/square_size)
                        assert np.max(label.as_list()) <= 1.0, f"Normalization did not work: {label.as_list()}\n  for file: {filename}"
                        label_file.write(f'0 {label.cx} {label.cy} {label.w} {label.h}\n')


    with open(output_dir / 'bounds.pkl', 'wb') as f:
        pickle.dump(bounds_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes test set out of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    parser.add_argument('--skip-empty', default=False, action='store_true',
        help='Whether to write images and annotation files that have nothing in them (noise)')
    parser.add_argument('--square-size', type=int, default=1024,
            help='image size to create')
    parser.add_argument('--output-dir', type=str,
            help='directory to store everything in (this script handles sub-directories)')
    parser.add_argument('--percent', type=float, default=0.1,
            help='normalized percent [0,1] of training set to use (random selection). ')
    args = parser.parse_args()
    main(args)