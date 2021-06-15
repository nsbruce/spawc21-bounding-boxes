import numpy as np
from rfinder.Config import Config
from sigmf import SigMFFile, sigmffile
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('TRAIN_DIR')
extension = '.sigmf-meta'

# I want to find the max bw, min bw, max t, min t in the dset
bws=[]
durations=[]

# For each sigmf metat file in the directory:
for i, item in enumerate(Path(input_dir).iterdir()):
    print(i)
    if not item.is_file():
        continue

    if item.suffix != extension:
        continue

    signal = sigmffile.fromfile(str(item.resolve()))

    for annotation in signal.get_annotations():
        duration = annotation[SigMFFile.LENGTH_INDEX_KEY]

        freq_start = annotation.get(SigMFFile.FLO_KEY)
        freq_stop = annotation.get(SigMFFile.FHI_KEY)
        bw = freq_stop-freq_start

        durations.append(duration)
        bws.append(bw)

results = np.empty((len(durations),2))

for i in range(results.shape[0]):
    results[i,0] = durations[i]
    results[i,1] = bws[i]

np.save('durs-bws.npy',results)


fig, ax = plt.subplots(2,1)
ax[0].hist(durations)
ax[0].set_title('Distribution of signal durations')
ax[1].hist(bws)
ax[1].set_title('Distribution of signal bandwidths')

plt.show()
