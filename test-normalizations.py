import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.stats import median_abs_deviation

import rfinder.data.converter as dc
import rfinder.plot.sigmf_utils as sigmfplt
from sigmf import sigmffile, SigMFFile
# from rfinder.config import Config


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


input_extension = '.sigmf-meta'
output_extension = '.png'

load_dotenv()  # take environment variables from .env.

eval_sigmf_dir = Path(os.getenv('EVAL_DIR'))
# rfind_sigmf_dir = Path(
#     '/project/def-msteve/nsbruce/RFI/ursi-data-processing/sigmfs')


NFFT = 1024
noverlap = NFFT//2

chosen_eval_stems = [
    # 'west-wideband-modrec-ex19-tmpl13-20.04',
    # 'west-wideband-modrec-ex5-tmpl12-20.04',
    # 'west-wideband-modrec-ex15-tmpl3-20.04'
    'west-wideband-modrec-ex95-tmpl10-20.04'
    # 'west-wideband-modrec-ex125-tmpl15-20.04'
]

# chosen_rfind_stems = [
#     'ens10f1_0_0.cap.x'
# ]

items = []
for stem in chosen_eval_stems:
    items.append(eval_sigmf_dir / (stem+input_extension))
# for stem in chosen_rfind_stems:
#     items.append(rfind_sigmf_dir / (stem+input_extension))

fig, ax = plt.subplots(1, len(items)+1, figsize=(6*(len(items)+1), 6))

for i, item in enumerate(items):

    filename = str(item.resolve())
    print(filename)

    signal = sigmffile.fromfile(filename)
    samples = signal.read_samples(start_index=0, count=signal.sample_count)
    fs = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    fc = signal.get_global_field(SigMFFile.FREQUENCY_KEY, 0)

    # with Timer('PSD'):
    #     fig.suptitle('PSDs')
    #     ax[i].set_title(item.stem)
    #     spec, freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=fs, Fc=fc, NFFT=NFFT, noverlap=noverlap, mode='psd')

    # with Timer('kurtosis'):
    #     fig.suptitle('kurtosis')
    #     ax[i].set_title(item.stem)
    #     spec, freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=fs, Fc=fc, NFFT=NFFT, noverlap=noverlap, mode='kurtosis', M=100)

    # with Timer('median'):
    #     fig.suptitle('median normalized')
    #     ax[i].set_title(item.stem)
    #     spec, freqs, times = dc.timeseries_to_waterfall(
    #         x=samples, Fs=fs, Fc=fc, NFFT=NFFT, noverlap=noverlap, mode='median normalized')
    with Timer('mean finder'):
        spec,freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=fs, Fc=fc, NFFT=NFFT, noverlap=noverlap, mode='psd')
        print(spec.shape)
        print(np.mean(spec[-25000:,:]))

    ax[i].imshow(spec[-25000:,:], aspect='auto', cmap='Greys',
                 interpolation='none',
                 origin='upper',
                 extent=(freqs[0], freqs[-1], times[-1], times[0]),
                 )

    ax[i].set_xlabel('Frequency')
    ax[i].set_ylabel('Time')

plt.show()
