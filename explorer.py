import numpy as np
from rfinder.config import Config
from sigmf import SigMFFile, sigmffile
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

import rfinder.data.converter as dc
import rfinder.types as rt


load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('TRAIN_DIR')

print(input_dir)

extension = '.sigmf-meta'

# I want to find the max bw, min bw, max t, min t in the dset
bws=[]
durations=[]
durations_normalized=[]

frequency_separations=[]

min_count = 999999999
max_count = 0

items = [Path(input_dir).glob('*'+extension)]

def overlapsInTime(boxA:rt.Box, boxB:rt.Box)->bool:
    boxA_lo = boxA.cy-boxA.h/2
    boxA_hi = boxA.cy+boxA.h/2

    boxB_lo = boxB.cy-boxB.h/2
    boxB_hi = boxB.cy+boxB.h/2

    if boxA_lo>boxB_hi or boxA_hi < boxB_lo:
        return True
    else:
        return False

    
# For each sigmf meta file in the directory:
for i, item in items:

    signal = sigmffile.fromfile(str(item.resolve()))
    boxes = dc.sigmf_to_boxes(str(item.resolve()))
    boxes.sort(key=lambda box: box.cx) # sorted by center frequency

    while len(boxes) > 0:
        box_lo = boxes[0]
        for k in range(1,len(boxes)):
            box_hi = boxes[k]
            # if 
            sep = (box_hi.cx-box_hi.w/2) - (box_lo.cx+box_lo.w/2)
            frequency_separations.append(sep)
        boxes.pop(0)

np.save('seps.npy', np.array(frequency_separations))

fig, ax = plt.subplots()
ax.hist(durations, bins=100)
ax.set_title('Distribution of signal durations')
plt.show()




#         annotations = signal.get_annotations()

#         min_count = min(len(annotations),min_count)
#         max_count = max(len(annotations),max_count)

#         for annotation in signal.get_annotations():
#             duration = annotation[SigMFFile.LENGTH_INDEX_KEY]
#             duration_norm = duration/signal.sample_count
#             if duration_norm > 1:
#                 print(item.name)
#                 break

#             freq_start = annotation.get(SigMFFile.FLO_KEY)
#             freq_stop = annotation.get(SigMFFile.FHI_KEY)
#             bw = freq_stop-freq_start

#             durations.append(duration)
#             durations_normalized.append(duration_norm)
#             bws.append(bw)


# print(f"Min: {min_count}")
# print(f"Max: {max_count}")

# durations_normalized = np.array(durations_normalized)
# print(f"Duration normalized mean: {durations_normalized.mean()}")
# print(f"Duration normalized max: {durations_normalized.max()}")
# print(f"Duration normalized min: {durations_normalized.min()}")

# results = np.empty((len(durations),2))


# for i in range(results.shape[0]):
#     results[i,0] = durations[i]
#     results[i,1] = bws[i]

# np.save('durs-bws.npy',results)


# fig, ax = plt.subplots(2,1)
# ax[0].hist(durations)
# ax[0].set_title('Distribution of signal durations')
# ax[1].hist(bws)
# ax[1].set_title('Distribution of signal bandwidths')

# plt.show()


