import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path
import argparse

import rfinder.plot.sigmf_utils as sigmfplt
from rfinder.config import Config


def main(args):
    load_dotenv()  # take environment variables from .env.

    input_dir = os.getenv('TRAIN_DIR')+'/'
    # input_dir = '/home/nsbruce/RFI/spawc21-bounding-boxes/eval-temp/'
    input_extension = '.sigmf-meta'
    # output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/eval-temp/'
    output_extension = '.png'
    output_dir = Path(args.output_dir)
    # input_dir = Path(args.input_dir)

    input_dir = Path(input_dir)
    config = Config()

    NFFT=args.fft_size
    noverlap=NFFT//2


    items = input_dir.glob('*'+input_extension)

    output_stems=list(output_dir.glob('*'+output_extension))
    output_stems = [output.stem for output in output_stems]

    for item in items:
        if item.stem in output_stems:
            continue

        filename = str(item.resolve())
        print(filename)

        fig, ax = plt.subplots(1,1,figsize=(6,6))

        sigmfplt.full_sigmf_ax(ax=ax, sigmf_fname=filename, NFFT=NFFT, noverlap=noverlap, show_labels=args.show_labels)

        plt.savefig(output_dir / (item.stem+output_extension))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots and saves waterfalls of sigmf files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    parser.add_argument('--fft-size', type=int, default=1024,
            help='FFT size to window with')
    parser.add_argument('--input-dir', type=str,
            help='directory with sigmfs')
    parser.add_argument('--output-dir', type=str,
            help='directory to store images in')
    parser.add_argument('--show-labels', action='store_true', default=False)
    args = parser.parse_args()
    main(args)