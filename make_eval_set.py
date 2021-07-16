import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path
from sigmf import sigmffile, SigMFFile
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle

import rfinder.data.converter as dc

"""
Easily run with:
quick-sbatch -t 4:0 -s 'source ~/Documents/dotfiles/ComputeCanada/env-setups/rfi.sh && source ~/envs/rfi/bin/activate && python ~/RFI/spawc21-bounding-boxes/make_eval_set.py'
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

    input_dir = os.getenv('EVAL_DIR')+'/'

    extension = '.sigmf-meta'

    items = Path(input_dir).glob('*'+extension)


    print(f"PREPARING TO BUILD SPLIT APART SQUARE FILES")
    output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/'
    square_size=1024

    bounds_dict=defaultdict()
    with open(output_dir+"eval.txt", "w") as eval_list_file:

        for item in items:
            filename = str(item.resolve())
            
            imgs, bounds_per_img = dc.sigmf_to_evaluation_images(filename, img_w=square_size,
                                        img_h=square_size, NFFT=square_size, noverlap=square_size//2,
                                        img_overlap=square_size//2)

            imgs_maxs = [np.max(img) for img in imgs]
            vmax = max(imgs_maxs)
            imgs_mins = [np.min(img) for img in imgs]
            vmin = min(imgs_mins)

            imgs_labels_iter = zip(imgs, bounds_per_img)
            for i in range(len(imgs)):
                img, bound = next(imgs_labels_iter)

                output_img_fname = item.stem+f'-{i}.png'
                plt.imsave(output_dir+'Images/'+output_img_fname, img, vmin=vmin, vmax=vmax)

                eval_list_file.write(f"\n{output_dir+output_img_fname}")

                bounds_dict[output_img_fname] = bound        

    with open(output_dir+'bounds.pkl', 'wb') as f:
        pickle.dump(bounds_dict, f)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes evaluation set out of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )

    args = parser.parse_args()
    main(args)