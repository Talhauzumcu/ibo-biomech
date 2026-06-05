#%%
from handlers import *
from biomech_io import *
import h5py
#%%
filename = '03_PRE_GANG12_01.c3d'
#%%
c3dh = C3DHandler(filename)
c3dh.load_data()
# %%
FileConverter.c3d_to_h5(filename, 'test_output.h5')
#%%
FileConverter.h5_to_trc('test_output.h5', 'test_output.trc')
#%%
FileConverter.c3d_to_trc(filename, 'test_output_direct.trc')
#%%
FileConverter.h5_to_mot('test_output.h5', 'test_output.mot')
#%%
FileConverter.h5_to_opensim('test_output.h5', 'test_output.mot', 'test_output.trc')
# %%
h5h = H5Handler('test_output.h5')
trialdata = h5h.load_data()
print(trialdata)
trialdata.forces['forceplate_0'].plot()
trialdata.forces['forceplate_0'].lowpass_filter(5)
trialdata.forces['forceplate_0'].plot()
#%%
# h5_file = h5py.File('test_output.h5', 'r')
# %%

