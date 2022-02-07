from pathlib import Path
from sigmf import sigmffile, SigMFFile
from dotenv import load_dotenv
import os


input_dir = './'
items = list(Path(input_dir).glob('*_nick.sigmf-meta'))
print(items)


for item in items:
     signal = sigmffile.fromfile(input_dir+item.name)
     dataname = item.name.split('_nick')[0]+'.sigmf-data'
     signal.set_data_file(dataname)
     signal.tofile(item.name)
     
    #  globalinfo = signal.get_global_info()
    #  captures = signal.get_captures()
    #  annotations = signal.get_annotations()
    #  meta = SigMFFile(data_file=dataname, global_info=globalinfo)
    #  for capture in captures:
    #      meta.add_capture(0,capture)
    #  for ann in annotations:
    #      meta.add_annotation(ann['core:sample_start'],ann['core:sample_count'], ann)
    #  meta.tofile(item.name.split('_nick')[0]+'_updated.sigmf-meta')

# load_dotenv()  # take environment variables from .env.
# input_dir = os.getenv('EVAL_DIR')+'/'
# sigmf_dir = os.getenv('TRAIN_DIR')+'/'

# for item in items:
#     signal = sigmffile.fromfile(input_dir+item.name)
#     print("file: "+signal.data_file)