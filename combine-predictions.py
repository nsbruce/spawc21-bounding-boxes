import numpy as np
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from sigmf import SigMFFile, sigmffile
import matplotlib.pyplot as plt
from typing import List

import rfinder.data.converter as dc
import rfinder.types as rt
from rfinder.config import Config
import rfinder.plot.utils as pu
import rfinder.data.metrics as dm

def numbered_fname_to_sigmf_fname(fname:str)->str:
    return fname.rsplit('-',1)[0]+'.sigmf-meta'


def merge_boxes(boxes:List[rt.Box])->List[rt.Box]:
    merged=[boxes[0]]
    boxes.remove(merged[0])

    old_len=-1
    while len(boxes) > 0:
        old_len = len(merged)

        for i, merged_box in enumerate(merged):
            del_idxs=[]
            for j, pred_box in enumerate(boxes):
                if dm.IOU(merged_box, pred_box) > 0:
                    merged_box.merge(pred_box)
                    merged[i] = merged_box
                    del_idxs.append(j)
            del_idxs.sort(reverse=True)
            for k in del_idxs:
                del boxes[k]
            if len(merged) == old_len and len(boxes) != 0:
                merged.append(boxes[0])
                del boxes[0]
    return merged


def main(args):
    if args.pred_dir != '/':
        pred_dir = args.pred_dir + '/'

    if args.output_dir != '/':
        output_dir = args.output_dir + '/'

    config = Config()

    load_dotenv()  # take environment variables from .env.
    if args.eval_set:
        sigmf_dir = os.getenv('EVAL_DIR')+'/'
    else:
        sigmf_dir = os.getenv('TRAIN_DIR')+'/'


    with open(args.bounds_file, 'rb') as pkl_file:
        bounds_dict = pickle.load(pkl_file)
    
    pred_txts = Path(pred_dir).glob('*.txt')
    sigmf_names=[]
    for pred_txt in pred_txts:
        sigmf_names.append(numbered_fname_to_sigmf_fname(pred_txt.stem))
    sigmf_names = set(sigmf_names)

    # #! HARD
    # sigmf_names=['west-wideband-modrec-ex3-tmpl8-20.04.sigmf-meta']

    for sigmf_name in sigmf_names:
        print(sigmf_name)

        signal = sigmffile.fromfile(sigmf_dir+sigmf_name)

        samples = signal.read_samples(start_index=0, count=signal.sample_count)
        fs = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        fc = signal.get_global_field(SigMFFile.FREQUENCY_KEY, 0)
        spec, freqs, times = dc.timeseries_to_waterfall(x=samples, NFFT=config.sigmf.nfft,
                                                        noverlap=config.sigmf.noverlap, Fs=fs, Fc=fc)

        df = freqs[1]-freqs[0]
        dt = times[1]-times[0]

        pred_boxes = []
        # Get sorted list of bounds
        current_stem = Path(sigmf_name).stem

        print('   getting boxes')
        
        for img_fname, img_bound in bounds_dict.items():
            if current_stem in img_fname:
                txt_fname = Path(img_fname).stem+'.txt'

                with open(pred_dir+txt_fname) as pred_txt:
                    for line in pred_txt.readlines():
                        _, cx, cy, w, h = line.split()
                        pred_box = rt.Box(conf=1, cx=float(cx), cy=float(cy), w=float(w), h=float(h))
                        # shift to where they exist actually
                        pred_box.scale(cy=1024, cx=1024, w=1024, h=1024)
                        pred_box.scale(cx=df, cy=dt, w=df, h=dt)
                        pred_box.shift(x=img_bound.cx-img_bound.w/2, y=img_bound.cy-img_bound.h/2)
                        # pred_box.scale(cy=0.5, h=200) #TODO what the hell
                        pred_boxes.append(pred_box)
                
        print(f'   got {len(pred_boxes)} boxes')
        print('   merging boxes')

        # merge all predictions from this scope that can be merged
        merged_boxes = merge_boxes(pred_boxes)        
        del pred_boxes
        print(f'   now have {len(merged_boxes)} boxes')

        print('   saving pickle')
        with open(output_dir+current_stem+'_preds.pkl', 'wb') as f:
            pickle.dump(merged_boxes, f)
        
        # get annotation boxes
        labels = dc.sigmf_to_boxes(sigmf_dir+sigmf_name)
        print(f'   have {len(labels)} boxes from true annotations')

        print('   plotting')
        # plot both box sets on top of waterfall
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        
        ax.imshow(spec, aspect='auto', cmap='viridis', 
            interpolation='none',
            origin='upper',
            extent=(freqs[0], freqs[-1], times[-1], times[0]),
        )

        ax.set_title(current_stem)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Time')

        print('labels')
        print(labels[0].as_list())
        print('merged_boxes')
        print(merged_boxes[0].as_list())

        pu.add_rect_patch(ax, boxes=labels, color=config.plotting.color['label'])
        pu.add_rect_patch(ax, boxes=merged_boxes, color=config.plotting.color['pred'])

        plt.savefig(output_dir+current_stem+'_pred.png')
        plt.close()

        # TODO write out boxes to sigmf annotations

        print()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combines darknet predictions from overlapping images, plots, and writes to sigmf annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    parser.add_argument('--pred-dir', type=str,
            help='directory containing the predictions (in darknet label format)')
    parser.add_argument('--bounds-file', type=str,
            help='pickle file containing bounds of each image')
    parser.add_argument('--eval-set', default=False, action='store_true',
        help='Which sigmf dataset to draw sigmf files from')
    parser.add_argument('--output-dir', type=str,
            help='directory to store output images and bounding boxes (pickles in)')
    args = parser.parse_args()
    main(args)