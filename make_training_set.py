import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path
from sigmf import sigmffile, SigMFFile
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split

import rfinder.data.converter as dc
from rfinder.config import Config

"""
Easily run with:
quick-sbatch -t 4:0 -s 'source ~/Documents/dotfiles/ComputeCanada/env-setups/rfi.sh && source ~/envs/rfi/bin/activate && python ~/RFI/spawc21-bounding-boxes/make_training_set.py --full'
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
    config = Config()

    extension = '.sigmf-meta'

    items = Path(input_dir).glob('*'+extension)



    if args.full:
        print(f"PREPARING TO BUILD FULL FILES")
        output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-full/'
        output_stems=list(Path(output_dir).glob('*.png'))
        output_stems = [output.stem for output in output_stems]
    else:
        print(f"PREPARING TO BUILD SPLIT APART SQUARE FILES")
        output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-square/'
        output_stems=list(Path(output_dir).glob('*.png'))
        output_stems = [output.stem for output in output_stems]
        square_size=1024



    for item in items:
        filename = str(item.resolve())

        if args.full and item.stem in output_stems:
            print(f"Skipping {item.stem}")
            continue

        if not args.full and item.stem+'-1' in output_stems and item.stem+'-189' not in output_stems:
            raise ValueError(f"don't have both sides of {item.stem} you need to clean up better")
        if not args.full and item.stem+'-1' in output_stems and item.stem+'-189' in output_stems:
            print(f"Skipping {item.stem}")
            continue

        if args.full:
            signal = sigmffile.fromfile(filename)
            samples = signal.read_samples(start_index=0, count=signal.sample_count)
            fs = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
            fc = signal.get_global_field(SigMFFile.FREQUENCY_KEY, 0)

            spec, freqs, times = dc.timeseries_to_waterfall(x=samples, NFFT=config.sigmf.nfft, noverlap=config.sigmf.noverlap, Fs=fs, Fc=fc)

            labels = dc.sigmf_to_boxes(filename)

            #* Skips empty images
            if args.skip_empty and len(labels) == 0:
                continue

            plt.imsave(output_dir+item.stem+'.png', spec)

            #* Need labels to be
            #* <object-class> <x_center> <y_center> <width> <height>
            with open (output_dir+item.stem+'.txt', 'a') as label_file:
                for i,label in enumerate(labels):
                    # frequencies are normalized so w is fine but cx needs to go from 0-1
                    labels[i] = label.shift(x=fs/2)
                    # y is in time so h and cy need to be normalized
                    # specgram doesn't end up with full time (factor of nfft and noverlap)
                    time_scale = np.array([times[-1], signal.sample_count/fs]).max() #TODO this will always be signal.sample_count/fs
                    labels[i] = label.scale(cy=1.0/time_scale, h=1.0/time_scale)

                    assert np.max(label.as_list()) <= 1.0, f"Normalization did not work: {label.as_list()}\n  for file: {filename}"

                    label_file.write(f'0 {label.cx} {label.cy} {label.w} {label.h}\n')
        
        else:
            imgs, labels_per_img, _ = dc.sigmf_to_labelled_images(filename, img_w=square_size, img_h=square_size,
                                                                NFFT=square_size, noverlap=square_size//2,
                                                                img_overlap=0)

            imgs_maxs = [np.max(img) for img in imgs]
            vmax = max(imgs_maxs)
            imgs_mins = [np.min(img) for img in imgs]
            vmin = min(imgs_mins)

            imgs_labels_iter = zip(imgs, labels_per_img)
            for i in range(len(imgs)):
                img, labels = next(imgs_labels_iter)

                # Skips empty images
                if args.skip_empty and len(labels)==0:
                    continue

                plt.imsave(output_dir+item.stem+f'-{i}.png', img, vmin=vmin, vmax=vmax)
                with open (output_dir+item.stem+f'-{i}.txt', 'a') as label_file:
                    for j, label in enumerate(labels):
                        labels[j] = label.scale(cy=1.0/square_size, cx=1.0/square_size, w=1.0/square_size, h=1.0/square_size)

                        assert np.max(label.as_list()) <= 1.0, f"Normalization did not work: {label.as_list()}\n  for file: {filename}"

                        label_file.write(f'0 {label.cx} {label.cy} {label.w} {label.h}\n')


        # output_txts = list(Path(output_dir).glob('*.txt'))

        # txt_sizes=defaultdict()
        # for output_txt in output_txts:
        #     txt_sizes[output_txt] = line_count(output_txt)

        


        # fig, ax = plt.subplots(2,1, figsize=(6,8))
        # ax[0].set_title('Before pruning 0s')
        # ax[1].set_title('After pruning 0s')
        # ax[0].set_xlabel('Number of boxes per file')
        # ax[0].set_ylabel('Number of files')
        # ax[1].set_xlabel('Number of boxes per file')
        # ax[1].set_ylabel('Number of files')
        # ax[0].hist(txt_sizes.values())

        # non_zeros = []
        # for key,val in txt_sizes.items():
        #     if val == 0:
        #         key.rename(str(key.parent)+'/empty/'+key.name)
        #         Path(str(key.parent)+'/'+key.stem+'.png').rename(str(key.parent)+'/empty/'+key.stem+'.png')
        #     else:
        #         non_zeros.append(val)

        
        # ax[1].hist(non_zeros)

        # plt.savefig('histograms.png')

        output_imgs = list(Path(output_dir+'/Images/').glob('*.png'))

        # train_imgs, test_imgs = train_test_split(output_imgs, test_size=0.2, train_size=0.8)


        with open(output_dir+"train.txt", "w") as trainfile:
            trainfile.write("\n".join(str(train_img.resolve()) for train_img in output_imgs ))

        # with open("test.txt", "w") as testfile:
        #     testfile.write("\n".join(str(test_img.resolve()) for test_img in test_imgs ))


        








if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes training set out of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )

    parser.add_argument('--full', default=False, action='store_true',
        help='Whether to make a full image for each sigmf file or not')
    parser.add_argument('--skip-empty', default=False, action='store_true',
        help='Whether to write images and annotation files that have nothing in them (noise)')
    args = parser.parse_args()
    main(args)