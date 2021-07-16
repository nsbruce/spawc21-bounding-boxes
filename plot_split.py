import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import numpy as np

import rfinder.data.converter as dc
import rfinder.plot.utils as pu
from rfinder.config import Config
from rfinder.plot.sigmf_utils import full_sigmf_ax

load_dotenv()  # take environment variables from .env.

input_dir = os.getenv('TRAIN_DIR')+'/'
config = Config()

# filename = 'west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta'
# filename = 'west-wideband-modrec-ex69-tmpl10-20.04.sigmf-meta'
filename='west-wideband-modrec-ex6-tmpl13-20.04.sigmf-meta'


NFFT=1024
img_w = 512
img_h = 512

imgs, labels = dc.sigmf_to_labelled_images(input_dir+filename, NFFT=NFFT, noverlap=NFFT//2, img_w=img_w, img_h=img_h)

imgs_maxs = [np.max(img) for img in imgs]
vmax = max(imgs_maxs)
imgs_mins = [np.min(img) for img in imgs]
vmin = min(imgs_mins)

img_label_iter = zip(imgs, labels)
color = config.plotting.color['label']

try:
    while True:
        fig, axs = plt.subplots(NFFT//img_w, NFFT//img_h)
        for ax in axs.reshape(-1):
            img, label = next(img_label_iter)
            pu.single_img_ax(ax=ax, img=img, vrange=(vmin,vmax))
            pu.add_center_dot_patch(ax, label, color)
            pu.add_rect_patch(ax, label, color)        

        plt.show()
        

except StopIteration:
    print("done all images")