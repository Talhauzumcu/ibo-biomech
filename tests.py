#%%
from containers import *
from handlers import *
from biomech_io import *
#%%
fileconverter = FileConverter()
fileconverter.c3d_to_h5('./P01_pre_gait_16_0001.c3d', 'test_output.h5')

#%%
h5h = H5Handler('test_output.h5')
trial_data = h5h.load_data()

# %%
