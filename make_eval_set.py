# import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
# from pathlib import Path
# from sigmf import sigmffile, SigMFFile
# import numpy as np
import argparse
# from collections import defaultdict
# from sklearn.model_selection import train_test_split
# import pickle
# from skimage import io
from typing import Optional


import rfinder.data.converter as dc

"""
Easily run with:
quick-sbatch -t 4:0 -s 'source ~/Documents/dotfiles/ComputeCanada/env-setups/rfi.sh && source ~/envs/rfi/bin/activate && python ~/RFI/spawc21-bounding-boxes/make_eval_set.py'
"""

# def _make_gen(reader):
#     b = reader(1024 * 1024)
#     while b:
#         yield b
#         b = reader(1024*1024)

# def line_count(filename):
#     f = open(filename, 'rb')
#     f_gen = _make_gen(f.raw.read)
#     return sum( buf.count(b'\n') for buf in f_gen )

# def main(args):

#     output_dir = Path(args.output_dir)

#     if not output_dir.exists():
#         raise FileNotFoundError (f"Output directory {output_dir} does not exist")

#     images_dir = output_dir / 'Images/'
#     if not images_dir.exists():
#         images_dir.mkdir()

#     load_dotenv()  # take environment variables from .env.

#     input_dir = os.getenv('EVAL_DIR')+'/'
#     extension = '.sigmf-meta'
#     items = list(Path(input_dir).glob('*'+extension))


#     print(f"PREPARING TO BUILD SPLIT APART SQUARE FILES")
#     square_size = args.square_size
#     NFFT = args.fft_size
#     if NFFT == -1:
#         NFFT = square_size


#     bounds_dict=defaultdict()
#     with open(output_dir / "eval.txt", "w") as eval_list_file:

#         for i, item in enumerate(items):
#             print(f' {i+1}/{len(items)}')

#             filename = str(item.resolve())
            
#             imgs, _, bounds_per_img = dc.sigmf_to_windowed_images(filename, img_w=square_size,
#                                         img_h=square_size, NFFT=NFFT, noverlap=NFFT//2,
#                                         img_overlap=square_size//2,
#                                         return_labels=False, mode='median normalized')

#             # imgs_maxs = [np.max(img) for img in imgs]
#             # vmax = max(imgs_maxs)
#             # imgs_mins = [np.min(img) for img in imgs]
#             # vmin = min(imgs_mins)

#             imgs_labels_iter = zip(imgs, bounds_per_img)
#             for j in range(len(imgs)):
#                 # Skips images that already exist
#                 output_img_path = images_dir / (item.stem+f'-{j}.png')
#                 if output_img_path.exists():
#                     continue

#                 img, bound = next(imgs_labels_iter)


#                 # plt.imsave(output_img_path, img, vmin=vmin, vmax=vmax)
#                 io.imsave(output_img_path, img.astype(np.uint16))

#                 bounds_dict[output_img_path.name] = bound        

#                 # output_img_fname = item.stem+f'-{j}.png'
#                 # plt.imsave(output_dir+'Images/'+output_img_fname, img, vmin=vmin, vmax=vmax)

#                 eval_list_file.write(f'{output_img_path.resolve()}\n')
#                 # eval_list_file.write(f"\n{output_dir+output_img_fname}")

#                 # bounds_dict[output_img_fname] = bound   

#     with open(output_dir / 'bounds.pkl', 'wb') as f:
#         pickle.dump(bounds_dict, f)


def main(args):

    load_dotenv()  # take environment variables from .env.

    sigmf_dir = os.getenv('EVAL_DIR')+'/'

    dc.sigmfs_to_label_image_files(sigmf_dir=sigmf_dir, output_dir=args.output_dir, img_size=args.square_size, NFFT=args.fft_size, for_training=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes evaluation set out of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    parser.add_argument('--square-size', type=int, default=1024,
            help='image size to create')
    parser.add_argument('--output-dir', type=str,
            help='directory to store everything in (this script handles sub-directories)')
    parser.add_argument('--fft-size', type=Optional[int], default=None,
            help='fft size to use. If not set will default to square-size.')
    args = parser.parse_args()
    main(args)