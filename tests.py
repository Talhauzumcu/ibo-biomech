
#%%
import sys
import os
# import h5py
import numpy as np
from pathlib import Path
from biomech_io.file_converter import FileConverter
#%%
c3dpath = str(Path('./P01_pre_gait_16_0001.c3d').resolve())
fileconverter = FileConverter()
fileconverter.c3d_to_h5(c3dpath, 'test_output_compressed.h5')

# %%
