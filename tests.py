#%%
from handlers import *
from biomech_io import *
import h5py
#%%
path = '/mnt/nas/phd/HTO/nmsm/nmsm_files/data/'
filename = '17_PRE_GANG_12_07.c3d'
#%%
c3dh = C3DHandler(path + filename)
c3dh.load_data()
# %%
FileConverter.c3d_to_h5(path + filename, 'test_output.h5')
#%%
FileConverter.h5_to_trc('test_output.h5', 'test_output.trc')
#%%
FileConverter.c3d_to_trc(path + filename, 'test_output_direct.trc')
#%%
FileConverter.h5_to_mot('test_output.h5', 'test_output.mot')
# %%
h5h = H5Handler('test_output.h5')
trialdata = h5h.load_data()
#%%
h5_file = h5py.File('test_output.h5', 'r')
# %%

