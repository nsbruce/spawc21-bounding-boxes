import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from sigmf import SigMFFile, sigmffile
import matplotlib.pyplot as plt
from typing import Dict, List
import itertools
import json
from collections import defaultdict

import rfinder.data.converter as dc
import rfinder.types as rt
from rfinder.config import Config
import rfinder.plot.utils as pu
import rfinder.data.metrics as dm


def numbered_fname_to_sigmf_fname(fname:str)->str:
    return fname.rsplit('-',1)[0]+'.sigmf-meta'


def merge_boxes(boxes:List[rt.Box])->List[rt.Box]:
    """ Brute force box merger """
    while (1):
        found = 0
        for boxA, boxB in itertools.combinations(boxes, 2):
            if dm.IOU(boxA, boxB)>0:
                newBox = boxA.merge(boxB)
                if boxA in boxes:
                    boxes.remove(boxA)
                if boxB in boxes:
                    boxes.remove(boxB)
                boxes.append(newBox)
                found = 1
                break
        if found == 0:
            break

    return boxes

def json_objects_to_boxes(objects:Dict, threshold:float)->List[rt.Box]:
    """
    Converts json objects dictionary into bounding boxes. This expects the json
    objects shape created by darknet.
    """
    boxes = []
    for pred in objects:
        conf = pred['confidence']
        if conf<threshold:
            continue
        cx = pred['relative_coordinates']['center_x'] 
        cy = pred['relative_coordinates']['center_y'] 
        w = pred['relative_coordinates']['width'] 
        h = pred['relative_coordinates']['height'] 
        boxes.append(rt.Box(conf,cx,cy,w,h))
    return boxes


def main(args):

    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError (f"Output directory {output_dir} does not exist")

    config = Config()

    load_dotenv()  # take environment variables from .env.
    if args.eval_set:
        sigmf_dir = os.getenv('EVAL_DIR')+'/'
    else:
        sigmf_dir = os.getenv('TRAIN_DIR')+'/'


    with open(args.bounds_file, 'rb') as pkl_file:
        bounds_dict = pickle.load(pkl_file)
    

    with open(args.results_json, 'r') as results_json:
        preds_json = json.load(results_json)

    preds_dict = defaultdict(defaultdict)
    for pred in preds_json:
        pred_fname = Path(pred['filename'])
        img_name = pred_fname.name
        sigmf_name = numbered_fname_to_sigmf_fname(img_name)
        img_boxes = json_objects_to_boxes(pred['objects'], args.threshold)
        try:
            preds_dict[sigmf_name][img_name] = img_boxes
        except KeyError:
            preds_dict[sigmf_name][img_name] = defaultdict()
            preds_dict[sigmf_name][img_name] = img_boxes
    
    isFirst=True
    for sigmf_name, imgs_dict in preds_dict.items():
        # if not args.eval_set and sigmf_name != 'west-wideband-modrec-ex3-tmpl8-20.04.sigmf-meta':
        #     continue
        # if args.eval_set and sigmf_name != 'wesst-wideband-modrec-ex15-tmpl3-20.04.sigmf-meta':
        #     continue

        print(sigmf_name)

        signal = sigmffile.fromfile(sigmf_dir+sigmf_name)

        if isFirst or args.plot:

            samples = signal.read_samples(start_index=0, count=signal.sample_count)
            fs = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
            fc = signal.get_global_field(SigMFFile.FREQUENCY_KEY, 0)
            spec, freqs, times = dc.timeseries_to_waterfall(x=samples, NFFT=config.sigmf.nfft,
                                                            noverlap=config.sigmf.noverlap, Fs=fs, Fc=fc)

            df = freqs[1]-freqs[0]
            dt = times[1]-times[0]

            isFirst=False

        pred_boxes = []
        print('   getting boxes')

        for img_fname, img_boxes in imgs_dict.items():
            # print(f"img_fname: {img_fname}")
            # print(f"img_bound: {img_bound.as_list()} ")
            img_bound = bounds_dict[img_fname]
            for img_box in img_boxes:
                # print(f"  img_box before: {img_box.as_list()}")
                img_box.scale(cy=args.square_size, cx=args.square_size, w=args.square_size, h=args.square_size)
                img_box.scale(cx=df, cy=dt, w=df, h=dt)
                img_box.shift(x=img_bound.cx-img_bound.w/2, y=img_bound.cy-img_bound.h/2)
                # print(f"  img_box after: {img_box.as_list()}")
                pred_boxes.append(img_box)

                
        print(f'   got {len(pred_boxes)} boxes from darknet annotations')
        print('   merging boxes')

        # merge all predictions from this scope that can be merged
        merged_boxes = merge_boxes(pred_boxes)        
        del pred_boxes

        print(f'   now have {len(merged_boxes)} boxes')
        print('   saving pickle')
        current_stem = Path(sigmf_name).stem
        with open(output_dir / (current_stem+'_preds.pkl'), 'wb') as f:
            pickle.dump(merged_boxes, f)
        
        if args.plot:
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

            pu.add_rect_patch(ax, boxes=labels, color=config.plotting.color['label'])
            pu.add_rect_patch(ax, boxes=merged_boxes, color=config.plotting.color['pred'])

            plt.savefig(output_dir / (current_stem+'_pred.png'))
            plt.close()

        # Write out boxes to sigmf annotations
        for box in merged_boxes:
            len_samples=int(box.h//fs)
            start_index=int(box.cy*fs-len_samples/2)
            f1=box.cx-box.w/2
            f2=box.cx+box.w/2

            if start_index<0:
                start_index=0
            if start_index+len_samples > signal.sample_count:
                len_samples = signal.sample_count-start_index
            if f1 < -0.5:
                f1=-0.5
            if f2 > 0.5:
                f2 = 0.5


            try:
                signal.add_annotation(
                    start_index=start_index,
                    length=len_samples,
                    metadata={
                        SigMFFile.FLO_KEY: f1,
                        SigMFFile.FHI_KEY: f2,
                        SigMFFile.AUTHOR_KEY: 'nick'
                    }
                )
            except AssertionError:
                print(f'COULD NOT ADD THIS ANNOTATION! {box.as_list()}')
        
        signal.tofile(output_dir / (Path(sigmf_name).stem+'_nick.sigmf-meta'))


        print()



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combines darknet predictions from overlapping images, plots, and writes to sigmf annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
    parser.add_argument('--results-json', type=str,
            help='path to json file containing the predictions (in darknet label format)')
    parser.add_argument('--bounds-file', type=str,
            help='pickle file containing bounds of each image')
    parser.add_argument('--eval-set', default=False, action='store_true',
        help='Which sigmf dataset to draw sigmf files from')
    parser.add_argument('--output-dir', type=str,
            help='directory to store output images and bounding boxes (pickles) in')
    parser.add_argument('--square-size', type=int, default=1024,
            help='image size to expect')
    parser.add_argument('--threshold', type=float, default=0.25,
            help='the confidence threshold predictions must meet')
    parser.add_argument('--plot', default=False, action='store_true',
            help='whether to save images for each combined image with boxes overlaid')
    args = parser.parse_args()
    main(args)