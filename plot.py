import numpy as np
from rfinder.Config import Config
from sigmf import SigMFFile, sigmffile
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('TRAIN_DIR')

filename = 'west-wideband-modrec-ex69-tmpl10-20.04.sigmf-meta'


signal = sigmffile.fromfile(input_dir+'/'+filename)
samples = signal.read_samples(start_index=0, count=signal.sample_count)

plt.specgram(x=samples,NFFT=1024, Fs=1, Fc=0)
plt.show()