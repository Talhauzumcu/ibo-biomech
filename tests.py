#%%
from ibo_biomech import C3DHandler, H5Handler, FileConverter, GaitAnalyzer
from ibo_biomech.utils.utils import *
#%%
filename = Path('test_c3d.c3d')
folder = Path('example_data')
#%%
c3dh = C3DHandler(str(folder / filename))
trial_c3d = c3dh.load_data()
# %%
FileConverter.c3d_to_h5(str(folder / filename), 'test_output.h5')
#%%
FileConverter.h5_to_trc('test_output.h5', 'test_output.trc')
#%%
FileConverter.c3d_to_trc(str(folder / filename), 'test_output_direct.trc')
#%%
FileConverter.h5_to_mot('test_output.h5', 'test_output.mot')
#%%
FileConverter.h5_to_opensim('test_output.h5', 'test_output.mot', 'test_output.trc')
#%%
FileConverter.c3d_to_opensim(str(folder / filename), 'test_output_direct.mot', 'test_output_direct.trc')
#%%
h5h = H5Handler('test_output.h5')
trialdata = h5h.load_data()
trialdata.attach_IK_results('test_output/test_output_IK.mot')
trialdata.attach_ID_results('example_data/inverse_dynamics.sto')
trialdata.crop('ik_results', 0, 100)
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
#%%
trialdata.forces['forceplate_0'].plot()
trialdata.forces['forceplate_0'].lowpass_filter(5)
trialdata.forces['forceplate_0'].plot()
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

h5_file = './test_output/test_output.h5'
setup_file = './setup_files/example_ID_setup.xml'
model = './test_output/example_model.osim'
trc_file = './test_output/example_ref.trc'
# %%
OsimHandler.run_ik(model_path=model, 
                   setup_file='./setup_files/example_IK_setup.xml', 
                   trc_file=trc_file, 
                   output_file='test_output_IK.mot', initial_time=0, final_time=1)
# %%
OsimHandler.run_scaling(model_path=model, 
                        setup_file='./setup_files/example_scale_setup.xml', 
                        move_markers = True, 
                        h5_file=h5_file, 
                        output_file='tets/test_output_scaled.osim', 
                        mass=70,
                        initial_time=0.5,
                        final_time=0.6)
# %%
external_forces_file = build_extloads(r_idx=1, output_file='test_output/test_output_external_loads2.xml', h5_file=h5_file)

# %%
OsimHandler.run_id(model_path=model,
                   setup_file=setup_file,
                   mot_file='./test_output/test_output_IK.mot',
                   external_loads_file=external_forces_file,
                   output_file='./test_output/test_output_id.sto',
                   initial_time=0,
                   final_time=1)

# %%
from ibo_biomech.utils.utils import *
from ibo_biomech import IKResults
filepath = Path('./test_output/test_output_IK.mot')
ikresults = IKResults(filepath=filepath)

#%%
from ibo_biomech import H5Handler, C3DHandler
h5Handler = H5Handler('./example_data/test_h5.h5')
trialdata= h5Handler.load_data()
trialdata.attach_IK_results('./test_output/test_output_IK.mot')
trialdata.ik_results.to_rad()
trialdata.ik_results.data['knee_angle_r'][:] = 0
from copy import deepcopy
new_data = deepcopy(trialdata.ik_results.data['knee_angle_r']+200)
trialdata.ik_results.add_column('knee_angle_r_modified', new_data.data)
h5Handler.save_data(trialdata, 'test_wIK.h5')
# # %%
# import ibo_biomech
# ibo_biomech.FileConverter.c3d_to_h5('./example_data/test_c3d.c3d', './example_data/test_h5.h5')
# # %%
#%%
from ibo_biomech import H5Handler, C3DHandler, IKResults
h5Handler = H5Handler('./example_data/test_h5.h5')
trialdata= h5Handler.load_data()
trialdata.attach_IK_results('./test_output/test_output_IK.mot')
trialdata.crop('ik_results', 50, 100)
h5Handler.save_data(trialdata, 'test_wIK_cropped.h5')
# %%
from ibo_biomech import H5Handler, C3DHandler, IKResults
h5Handler = H5Handler('./example_data/test_h5.h5')
trialdata = h5Handler.load_data()
trialdata.attach_IK_results('./test_output/test_output_IK.mot')
h5Handler.save_data(trialdata, 'test_wIK.h5')
#%%
from ibo_biomech import H5Handler, C3DHandler, IKResults
h5Handler = H5Handler('test_wIK.h5')
trialdata = h5Handler.load_data()
trialdata.crop('ik_results', 50, 100)
h5Handler.save_data(trialdata, 'test_wIK_cropped.h5')
#%%
import ibo_biomech
ibo_biomech.FileConverter.c3d_to_h5('./example_data/test_c3d.c3d', './example_data/test_h5.h5')
#%%
import ibo_biomech
h5Handler = ibo_biomech.H5Handler('./example_data/test_h5.h5')
trialdata = h5Handler.load_data()
trialdata.crop('markers',50,100)
trialdata.crop('forces',100,1000)
new_h5 = h5Handler.save_data(trialdata, 'test_h5_cropped.h5')
ibo_biomech.FileConverter.h5_to_opensim(new_h5, 'test_h5_cropped.mot', 'test_h5_cropped.trc')
# %%
