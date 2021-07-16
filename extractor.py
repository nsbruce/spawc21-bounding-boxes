import numpy as np
from rfinder.config import Config
from sigmf import SigMFFile, sigmffile
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from scipy.signal import remez, lfilter

import rfinder.data.converter as dc
import rfinder.types as rt
from rfinder.plot.sigmf_utils import full_sigmf


load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('TRAIN_DIR')

#? USED TO FIND A FILE CONTAINING FM
find_fm = False
if find_fm:
    print(input_dir)

    extension = '.sigmf-meta'

    items = list(Path(input_dir).glob('*'+extension))

    # items = [str(item.resolve()) for item in items]

        
    # For each sigmf meta file in the directory:
    for item in items:
        signal = sigmffile.fromfile(str(item.resolve()))
        annotations = signal.get_annotations()
        for i,annotation in enumerate(annotations):
            if annotation['core:description'] == 'FM':
                print(f"FOUND AN FM SIGNAL IN FILE: {item.name}")
                print(f"ANNOTATION NO.: {i}")
                break

#? USED TO EXTRACT THE TIMESERIES
fm_file='west-wideband-modrec-ex6-tmpl13-20.04.sigmf-meta'
annotation_idx=1

print(f"FILE: {fm_file}")

fm_file = input_dir+'/'+fm_file
signal = sigmffile.fromfile(fm_file)

print(f" -| SIGMF DATATYPE: {signal.get_global_field(SigMFFile.DATATYPE_KEY)}")
annotation=signal.get_annotations()[annotation_idx]

sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
annotation_start_idx = annotation[SigMFFile.START_INDEX_KEY]
annotation_length = annotation[SigMFFile.LENGTH_INDEX_KEY]
print(f" -| ANNOTATION start: {annotation_start_idx/sample_rate} s")
print(f" -| ANNOTATION duration: {annotation_length/sample_rate} s")

freq_start = annotation.get(SigMFFile.FLO_KEY)
freq_stop = annotation.get(SigMFFile.FHI_KEY)
print(f" -| ANNOTATION fLO: {freq_start}")
print(f" -| ANNOTATION fHI: {freq_stop}")
bw=freq_stop-freq_start
print(f" -| ANNOTATION BW: {bw}")

samples = signal.read_samples(annotation_start_idx, annotation_length)

print(f" -| ANNOTATION mean(real): {np.real(samples).mean()}")
print(f" -| ANNOTATION mean(imag): {np.imag(samples).mean()}")
plot_full=False
if plot_full:
    print(f" -| Plotting full SigMF file")
    full_sigmf(fm_file,show_labels=True)

plot_cropped_time_full_bw=False
if plot_cropped_time_full_bw:
    spec, freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=sample_rate)
    print(f" -| Plotting full BW but cropped time for annotation")

    fig,ax = plt.subplots()
    ax.imshow(spec, aspect='auto')
    plt.show()

fc = (freq_stop+freq_start)/2
t_arr = np.arange(0,len(samples)*sample_rate,sample_rate**-1)
print(f" -| Shifting annotation to 0 (shifting by {-fc})")
samples *= np.exp(1j*2*np.pi*-fc*t_arr)

plot_cropped_time_full_bw_shifted=False
if plot_cropped_time_full_bw_shifted:
    spec, freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=sample_rate)
    print(f" -| Plotting full BW (shifted so that signal is centered) but cropped time for annotation")

    fig,ax=plt.subplots()
    ax.imshow(spec, aspect='auto')
    plt.show()

print(f" -| Making taps")
pb = np.ceil(bw*10/2)/10
taps=remez(50,[0, pb, pb+0.05, 0.5], [1,0])
print(f" -| Filtering to the center {2*pb} normalized Hz")
samples = lfilter(taps,1,samples)

plot_filtered=False
if plot_filtered:
    spec, freqs, times = dc.timeseries_to_waterfall(x=samples, Fs=sample_rate)
    print(f" -| Plotting filtered annotation")

    fig,ax=plt.subplots()
    ax.imshow(spec, aspect='auto')
    plt.show()

print(f" -| Demodulating annotation")
demodded = samples*np.conj(np.roll(samples,1))
demodded = np.angle(demodded)

plot_demodded=True
if plot_demodded:
    fig, ax = plt.subplots(2,1)
    fig.suptitle('Demodulated FM')
    ax[0].set_title('Time domain')
    ax[0].plot(demodded)
    ax[1].set_title('Frequency domain')
    ax[1].plot(np.fft.fftshift(np.fft.fft(demodded)))
    plt.show()