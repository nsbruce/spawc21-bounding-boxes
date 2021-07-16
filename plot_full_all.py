import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path

import rfinder.data.converter as dc
import rfinder.plot.sigmf_utils as sigmfplt
from rfinder.config import Config

load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('EVAL_DIR')+'/'
input_extension = '.sigmf-meta'
output_dir = '/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/full-waterfalls-eval/'
output_extension = '.png'
config = Config()

NFFT=1024
noverlap=512


items = Path(input_dir).glob('*'+input_extension)

output_stems=list(Path(output_dir).glob('*'+output_extension))
output_stems = [output.stem for output in output_stems]

for item in items:
    if item.stem in output_stems:
        continue

    filename = str(item.resolve())

    fig, ax = plt.subplots(1,1,figsize=(6,6))

    sigmfplt.full_sigmf_ax(ax=ax, sigmf_fname=filename, NFFT=NFFT, noverlap=noverlap, show_labels=True)

    plt.savefig(output_dir+item.stem+output_extension)
    plt.close()
