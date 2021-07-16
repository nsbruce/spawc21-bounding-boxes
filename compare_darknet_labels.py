from pathlib import Path
import numpy as np
import pickle

import rfinder.types as rt
import rfinder.data.metrics as dm

true_labels_list = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-square/test.txt'
true_labels_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-square/labels/'
pred_labels_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-square/Images/'

with open(true_labels_list, 'r') as true_list_file:
    true_img_paths = true_list_file.readlines()

label_names = list(Path(labelpath).stem+'.txt' for labelpath in true_img_paths)

IOUs=[]
true_box_counter=0

for label_name in label_names:
    true_file = true_labels_dir+label_name
    pred_file = pred_labels_dir+label_name

    true_boxes=[]
    with open(true_file, 'r') as truefile:
        for line in truefile.readlines():
            _, cx, cy, w, h = line.split()
            true_boxes.append(rt.Box(conf=1, cx=float(cx), cy=float(cy), w=float(w), h=float(h)))

    pred_boxes=[]
    with open(pred_file, 'r') as predfile:
        for line in predfile.readlines():
            _, cx, cy, w, h = line.split()
            pred_boxes.append(rt.Box(conf=1, cx=float(cx), cy=float(cy), w=float(w), h=float(h)))
    
    for true_box in true_boxes:
        temp_IOUs=[]
        for pred_box in pred_boxes:
            IOU = dm.IOU(true_box, pred_box)
            if IOU > 0:
                temp_IOUs.append(IOU)
        if len(temp_IOUs)==0:
            IOUs.append(-1)
            print("-1")
        else:
            IOUs.append(temp_IOUs)
            print(temp_IOUs)

    print()

with open('IOUs.pkl', 'wb') as f:
    pickle.dump(IOUs, f)


print(f"Number of true boxes: {len(IOUs)}")
empty_boxes = [item for item in IOUs if item==-1]
print(f"Number of false negatives (misses): {len(empty_boxes)} ({len(empty_boxes)/len(IOUs)*100}% of total true boxes)")
not_empty_boxes = [item for item in IOUs if item != -1]
single_IOUs = [item for item in not_empty_boxes if len(item)==1]
print(f"Number of true boxes with single prediction (IOU>0): {len(single_IOUs)} ({len(single_IOUs)/len(IOUs)*100}%)")
multiple_IOUs = [item for item in not_empty_boxes if len(item) >1]
print(f"Number of true boxes with multiple predictions (IOU>0): {len(multiple_IOUs)} ({len(multiple_IOUs)/len(IOUs)*100}%)")

print(f"Mean of all IOUs: {np.mean([item for sublist in not_empty_boxes for item in sublist])}")
print(f"Mean of max IOU for each true box: {np.mean([max(item) for item in not_empty_boxes])}")
