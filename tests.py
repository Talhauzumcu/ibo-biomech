#%%
from ibo_biomech import C3DHandler, H5Handler, FileConverter
#%%
filename = '06_PRE_GANG_12_15.c3d'
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
#%%
FileConverter.c3d_to_opensim(filename, 'test_output_direct.mot', 'test_output_direct.trc')
#%%
h5h = H5Handler('test_output.h5')
trialdata = h5h.load_data()
# print(trialdata)
# trialdata.forces['forceplate_0'].plot()
# trialdata.forces['forceplate_0'].lowpass_filter(5)
# trialdata.forces['forceplate_0'].plot()
#%%
# h5_file = h5py.File('test_output.h5', 'r')
# %%

#%%
from ibo_biomech import C3DHandler, H5Handler, FileConverter
from ibo_biomech.handlers.osimHandler import OsimHandler


h5_file = 'test_output.h5'
setup_file = 'example_IK_setup.xml'
model = 'example_model.osim'
trc_file = 'test_output.trc'
# %%
OsimHandler.run_ik(model_path=model, setup_file=setup_file, trc_file=trc_file, output_file='test_output/test_output_OW_Ik.mot', initial_time=0, final_time=1)
# %%
