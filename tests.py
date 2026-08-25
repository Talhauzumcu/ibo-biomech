#%%
from ibo_biomech import C3DHandler, H5Handler, FileConverter, GaitAnalyzer
from ibo_biomech.utils.utils import *
#%%
filename = Path('06_PRE_GANG_12_15.c3d')
folder = Path('example_data')
#%%
c3dh = C3DHandler(str(folder / filename))
trial_c3d = c3dh.load_data()
# %%
FileConverter.c3d_to_h5(str(folder / filename), 'test_output.h5')
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
trialdata.attach_IK_results('example_data/IKResults.mot')
trialdata.attach_ID_results('example_data/inverse_dynamics.sto')
trialdata.crop(0,100)
h5h.save_data(trialdata, 'test_output_with_results.h5')
#%%
bodyweight = 879
marker_re = 'R_ToesTop'
marker_li = 'L_ToesTop'
GaitAnalyzer.get_plate_contacts(bodyweight=bodyweight, 
                                trialdata=trialdata, 
                                marker_re=marker_re, 
                                marker_li=marker_li,
                                threshold_multiplier=1)
# print(trialdata)
# trialdata.forces['forceplate_0'].plot()
# trialdata.forces['forceplate_0'].lowpass_filter(5)
# trialdata.forces['forceplate_0'].plot()
#%%
EMG_channels = np.arange(0, 16)
trialdata.parse_EMG_data(EMG_channels)
#%%
h5h.save_data(trialdata, 'test_output_with_emg.h5')
#%%
# h5_file = h5py.File('test_output.h5', 'r')

#%%
from ibo_biomech import C3DHandler, H5Handler, FileConverter
from ibo_biomech.handlers.osimHandler import OsimHandler
from ibo_biomech.utils.utils import *

h5_file = 'test_output.h5'
setup_file = 'example_ID_setup.xml'
model = 'example_model.osim'
trc_file = 'example_ref.trc'
# %%
OsimHandler.run_ik(model_path=model, 
                   setup_file='example_IK_setup.xml', 
                   trc_file='test_output.trc', 
                   output_file='test_output/test_output_IK.mot', initial_time=0, final_time=1)
# %%
OsimHandler.run_scaling(model_path=model, 
                        setup_file='example_scale_setup.xml', 
                        move_markers = True, 
                        h5_file=h5_file, 
                        output_file='test_output/test_output_scaled.osim', 
                        mass=70,
                        initial_time=0.5,
                        final_time=0.6)
# %%
external_forces_file = build_extloads(r_idx=1, output_file='test_output/test_output_external_loads.xml', h5_file=h5_file)

# %%
OsimHandler.run_id(model_path=model,
                   setup_file=setup_file,
                   mot_file='test_output/test_output_IK.mot',
                   external_loads_file=external_forces_file,
                   output_file='test_output/test_output_id.sto',
                   initial_time=0,
                   final_time=1)

# %%
from ibo_biomech.utils.utils import *
from ibo_biomech import IKResults
filepath = Path('./test_output/test_output_IK.mot')
ikresults = IKResults(filepath=filepath)