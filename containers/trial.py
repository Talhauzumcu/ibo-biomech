from typing import Dict, List, Optional, Any
from handlers.c3dHandler import C3DHandler
from handlers.osimHandler import OsimHandler
from containers import AnalogData, ForceData, MarkerData
from functools import cached_property
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from utils import *
from copy import deepcopy

class Trial:    
    TRIAL_TYPES = ['gang','rotation','stand','stuhl_ab','stuhl_auf','treppen_ab','treppen_auf']
    
    def __init__(self, name: str, subject_id: str, filepath: str, metadata: List[str] = None,
                 c3d_handler: Optional[C3DHandler] = None, osim_handler: Optional[OsimHandler] = None):
        self.name = name
        self.subject_id = subject_id
        self.filepath = filepath
        self.metadata = metadata if metadata is not None else []
        self.c3d_handler = c3d_handler
        self.osim_handler = osim_handler
        self.c3d_data_loaded = False
        self.type = 'unknown'
        self._parse_metadata()
        self.osim_handler.attach_trial(self)

    def _parse_metadata(self):
        """parse metadata."""
        self.metadata = [data.lower() for data in self.metadata]  # Convert to lowercase for easier comparison
        self.timepoint = 'pre' if 'pre' in self.metadata else 'post' if 'post' in self.metadata else 'unknown'
        self.folderside = 'links' if 'links' in self.metadata else 'rechts' if 'rechts' in self.metadata else 'unknown'
        self.leg = 'R' if 'r' in self.metadata else 'L' if 'l' in self.metadata else 'unknown'
        for data in self.metadata:
            for trial_type in self.TRIAL_TYPES:
                if trial_type == data.lower():
                    self.type = trial_type
                    break

    def prepare_osim_files(self, output_path):
        """Prepare .mot and .trc files from c3d files."""
        fullpath = Path(output_path) / self.subject_id/ self.filepath.split(self.subject_id,1)[1].replace('.c3d','').lstrip('/')
        if self.type == 'stand':
            self.c3d_handler.prepare_virtual_c3d_files(fullpath)
        else:
            self.c3d_handler.prepare_osim_files(fullpath)

    def lowpass_filter(self, cutoff: int = 15) -> None:
        """Apply lowpass filter to the trial's C3D data."""
        self.c3d_handler.lowpass_filter_all(cutoff=cutoff)

    def slice_c3d(self, start_frame: int, end_frame: int) -> None:
        """Slice the trial's C3D data between start_frame and end_frame."""
        self.c3d_handler.slice_c3d(start_frame, end_frame)
        # Update the loaded data in the trial to reflect the sliced data in the handler.
        self.markers = self.c3d_handler.markers
        self.forces = self.c3d_handler.forces
        self.analogs = self.c3d_handler.analogs
    
    def write_c3d(self, output_path: str) -> None:
        """Write the trial's C3D data to a new file."""
        self.c3d_handler.write_c3d(output_path)

    def load_c3dData(self) -> None:
        """Load C3D data for the trial."""
        self.c3d_handler.load_data()
        self.c3d_data_loaded = True
        #Save the motion data to the trial level for native python access (c3ddata handler loads the ezc3d library which is not possible to pickle)
        self.markers = self.c3d_handler.markers
        self.forces = self.c3d_handler.forces
        self.analogs = self.c3d_handler.analogs

    def runIK(self) -> None:
        model_path = Path('./models/scaled_models').resolve() / f'P{self.subject_id}_{self.timepoint}.osim'
        trc_path = Path(self.osimPath).resolve() / f'{self.name}.trc'
        mot_path = Path(self.osimPath).resolve() /  f'{self.name}.mot'
        self.osim_handler.runIK(model_path=model_path, trc_path=trc_path, mot_path=mot_path)

    def runScaling(self, mass: float = None) -> None:
        '''Should be used only for the reference trial, which is used for scaling.'''
        if self.type != 'stand':
            print(f"Warning: Running scaling for a non-reference trial ({self.name}). This is not recommended.")

        model_path = Path('./models').resolve() / f'RajagopalLaiUhlrich2023_3DOFKnee_markered.osim'
        trc_path = Path(self.osimPath).resolve() / f'{self.name}.trc'
        output = Path('./models').resolve() / 'scaled_models' / f'P{self.subject_id}_{self.timepoint}.osim'
        self.osim_handler.runScaling(model_path=model_path, trc_path=trc_path, output=output, mass=mass)

    def get_IK_errors(self) -> Optional[Dict[str, np.ndarray]]:
        """Get IK errors from the log file."""
        log_file = Path(self.osimPath).resolve() / 'IKResults' / f'{self.name}_IK.log'
        if not log_file.exists():
            print(f"Log file {log_file} does not exist. Cannot retrieve IK errors.")
            return None
        
        errors = {}
        total_squared_errors = []
        RMS_errors = []
        max_errors = []
        with open(log_file, 'r') as f:
            for line in f:
                if "Frame" in line:
                    info = line.split('total')[-1].strip().split(',')
                    total_squared_error = float(info[0].split('=')[1].strip())
                    RMS_error = float(info[1].split('=')[1].strip())
                    max_error = float(info[2].split('=')[1].strip().split(' ')[0].strip())
                    total_squared_errors.append(total_squared_error)
                    RMS_errors.append(RMS_error)
                    max_errors.append(max_error)

        errors['total_squared_errors'] = np.array(total_squared_errors)
        errors['RMS_errors'] = np.array(RMS_errors)
        errors['max_errors'] = np.array(max_errors)
        return errors
    
    def validate_IK_results(self, threshold: float = 0.02) -> bool:
        """Validate IK results based on RMS error threshold."""
        if self.IK_errors is None:
            # print(f"Cannot validate IK results for trial {self.name} as IK errors could not be retrieved.")
            return False
        
        max_RMS_error = np.max(self.IK_errors['RMS_errors'])
        if max_RMS_error > threshold:
            # print(f"Warning: Maximum RMS error {max_RMS_error:.2f} exceeds the threshold of {threshold:.2f} for trial {self.name}.")
            return False
        return True

    def set_osimPath(self, osimpath: str) -> None:
        """Set the path for the osim files in the osim handler."""
        fullpath = Path(osimpath) / self.subject_id / self.filepath.split(self.subject_id,1)[1].replace('.c3d','').replace(self.name,'').lstrip('/')
        self.osimPath = fullpath

    def get_gait_cycle_indices(self) -> Optional[Dict[str, Any]]:
        #Assumes subject starts from the forceplate_0.
        fp = deepcopy(self.c3d_handler.forces['forceplate_0']) #copy the forceplate data to avoid modifying the original data in the handler
        downsampling_ratio = fp.sampling_rate / self.c3d_handler.markers['R_ToesTop'].sampling_rate
        fp.downsample(int(downsampling_ratio))
        vertical_forces = fp.Fz
        threshold = 20 #heel strike threshold
        above_threshold = vertical_forces > threshold
        if not np.any(above_threshold):
            print(f"Warning: No heel strike detected in trial {self.name}. Cannot determine gait cycle indices.")
            return None
        start_index = np.where(np.diff(above_threshold.astype(int)) == 1)[0][0] + 1
        leg = 'R' if self.c3d_handler.markers['R_ToesTop'].x[start_index] > self.c3d_handler.markers['L_ToesTop'].x[start_index] else 'L'
        fp2 = deepcopy(self.c3d_handler.forces['forceplate_1'])
        fp2.downsample(int(downsampling_ratio))
        vertical_forces_fp2 = fp2.Fz
        second_fp_start = np.where(np.diff((vertical_forces_fp2 > threshold).astype(int)) == 1)[0][0] + 1
        step_frames = (second_fp_start - start_index)
        end_index = start_index + 2 * step_frames
        if end_index > len(vertical_forces):
            print(f"Warning: Calculated end index {end_index} exceeds the length of the force data for trial {self.name}. Adjusting end index to the length of the data.")
            end_index = len(vertical_forces) - 1
        # second_foot_x = self.c3d_handler.markers[f'{leg}_HeelTop'].x[second_fp_start:]
        # second_foot_z = self.c3d_handler.markers[f'{leg}_HeelTop'].z[second_fp_start:]
        # second_foot_x_acc = np.diff(second_foot_x, n=2) #Calculate the second derivative of the foot x position to get acceleration.
        # second_foot_z_acc = np.diff(second_foot_z, n=2) #Calculate the second derivative of the foot z position to get acceleration.
        # second_foot_acc = np.sqrt(second_foot_x_acc**2 + second_foot_z_acc**2) #Calculate the magnitude of the foot acceleration.
        # end_index_candidates = find_peaks(second_foot_acc, height=2, prominence=1)[0]
        # if len(end_index_candidates) == 0:
        #     print(f"Warning: No second heelstrike detected for {leg} in trial {self.name}. Cannot determine gait cycle end index returning -1.")
        #     return {'start_index': start_index, 'end_index': -1, 'leg': leg}
        # end_index = end_index_candidates[0] + second_fp_start
        return {'start_index': start_index, 'end_index': end_index, 'leg': leg}

    def get_stance_phase_indices(self) -> Optional[Dict[str, Any]]:
        #Assumes subject starts from the forceplate_0.
        results = {}
        fp = deepcopy(self.c3d_handler.forces['forceplate_0']) #copy the forceplate data to avoid modifying the original data in the handler
        downsampling_ratio = fp.sampling_rate / self.c3d_handler.markers['R_ToesTop'].sampling_rate
        fp.downsample(int(downsampling_ratio))
        vertical_forces = fp.Fz
        threshold = 20 #heel strike threshold
        above_threshold = vertical_forces > threshold
        if not np.any(above_threshold):
            print(f"Warning: No heel strike detected in trial {self.name}. Cannot determine stance phase indices.")
            return None
        
        start_index_candidates = np.where(np.diff(above_threshold.astype(int)) == 1)
        end_index_candidates = np.where(np.diff(above_threshold.astype(int)) == -1)
        if len(start_index_candidates[0]) == 0:
            print(f"Warning: No heel strike detected in trial {self.name}. Cannot determine stance phase start index returning -1.")
            leg_1 = 'R' if self.c3d_handler.markers['R_ToesTop'].x[0] > self.c3d_handler.markers['L_ToesTop'].x[0] else 'L'
            results[leg_1] = {'start_index': -1, 'end_index': -1}
            start_index = -1
        else:
            start_index = start_index_candidates[0][0] + 1
        end_index = end_index_candidates[0][0] + 1

        leg_1 = 'R' if self.c3d_handler.markers['R_ToesTop'].x[start_index] > self.c3d_handler.markers['L_ToesTop'].x[start_index] else 'L'
        
        fp2 = deepcopy(self.c3d_handler.forces['forceplate_1'])
        fp2.downsample(int(downsampling_ratio))
        vertical_forces_fp2 = fp2.Fz
        above_threshold_fp2 = vertical_forces_fp2 > threshold
        start_index_fp2 = np.where(np.diff(above_threshold_fp2.astype(int)) == 1)[0][0] + 1
        end_index_fp2_candidates = np.where(np.diff(above_threshold_fp2.astype(int)) == -1)
        results[leg_1] = {'start_index': start_index, 'end_index': end_index}
        leg_2 = 'L' if leg_1 == 'R' else 'R'
        if len(end_index_fp2_candidates[0]) == 0:
            print(f"Warning: No toe off detected for {leg_2} in trial {self.name}. Cannot determine stance phase end index for {leg_2} returning -1.")
            results[leg_2] = {'start_index': start_index_fp2, 'end_index': -1}
            return results
        end_index_fp2 = end_index_fp2_candidates[0][0] + 1
        results[leg_2] = {'start_index': start_index_fp2, 'end_index': end_index_fp2}
        return results

    def __str__(self):
        return f"Trial Name: {self.name}, Type: {self.type},Leg: {self.leg}, Timepoint: {self.timepoint}, Folder Side: {self.folderside}"
    
    def __repr__(self):
        return f"Trial(name={self.name}, type={self.type}, leg={self.leg}, timepoint={self.timepoint}, folderside={self.folderside})"
    
    @cached_property
    def IK_errors(self) -> Optional[Dict[str, np.ndarray]]:
        """Property to get IK errors."""
        return self.get_IK_errors()

    @cached_property
    def valid_IK(self) -> Optional[bool]:
        """Property to check if IK results are valid."""
        return self.validate_IK_results()
    
    @cached_property
    def IK_results(self) -> pd.DataFrame:
        """Property to get IK results as a DataFrame."""
        return self.osim_handler.get_ik_dataframe()
    
    @cached_property
    def ID_results(self) -> pd.DataFrame:
        """Property to get ID results as a DataFrame."""
        return self.osim_handler.get_id_dataframe()
    
    @cached_property
    def IK_results_gait_cycle(self) -> pd.DataFrame:
        """Property to get IK results for the gait cycle as a DataFrame."""
        return self._get_IK_results_gait_cycle()
    
    @cached_property
    def IK_results_stance_phase(self) -> pd.DataFrame:
        """Property to get IK results for the stance phase as a DataFrame."""
        return self._get_IK_results_stance_phase()
    
    @cached_property
    def GRF_df(self) -> pd.DataFrame:
        """Property to get ground reaction force data as a DataFrame."""
        df = pd.DataFrame()
        if self.forces is None:
            print(f"Cannot create GRF DataFrame for trial {self.name} as force data is not loaded.")
            return None
        for i, fp in enumerate(self.forces.values()):
            time = np.arange(fp.force.shape[0]) / fp.sampling_rate if fp.sampling_rate else np.arange(fp.force.shape[0])
            df_ = pd.DataFrame(np.hstack([fp.force,fp.moment,fp.cop]), columns=[f'Fx_{i+1}', f'Fy_{i+1}', f'Fz_{i+1}', f'Mx_{i+1}', f'My_{i+1}', f'Mz_{i+1}', f'cop_x_{i+1}', f'cop_y_{i+1}', f'cop_z_{i+1}'])
            df = pd.concat([df, df_], axis=1)
            df['time'] = time
        return df
    
    def _get_IK_results_gait_cycle(self) -> pd.DataFrame:
        """Helper method to extract IK results for the gait cycle as a DataFrame."""
        #Need c3d data to get indices for the gait cycle.
        if not hasattr(self, '_IK_results_gait_cycle'):
            try:
                gait_cycle_indices = self.get_gait_cycle_indices()
                if gait_cycle_indices is None:
                    print(f"Cannot extract IK results for gait cycle in trial {self.name} due to missing gait cycle indices.")
                    return None
                start_idx, end_idx, lead_leg = gait_cycle_indices['start_index'], gait_cycle_indices['end_index'], gait_cycle_indices['leg']
                self._IK_results_gait_cycle = self.IK_results.iloc[start_idx:end_idx].reset_index(drop=True)
                self._IK_results_gait_cycle = time_normalize_df(self._IK_results_gait_cycle)
                self._IK_results_gait_cycle['lead_leg'] = lead_leg
                self.lead_leg = lead_leg # This is not a place to set this value. If necessary gotta change it in the future.
            except Exception as e:
                print(f"Error occurred while extracting IK results for gait cycle in trial {self.name}: {e}")
                return None
        return self._IK_results_gait_cycle
    
    def _get_IK_results_stance_phase(self) -> pd.DataFrame:
        """Helper method to extract IK results for the stance phase as a DataFrame."""
        #Need c3d data to get indices for the gait cycle and determine stance phase.
        if not hasattr(self, '_IK_results_stance_phase'):
            try:
                stance_phase_indices = self.get_stance_phase_indices()
                if stance_phase_indices is None:
                    print(f"Cannot extract IK results for stance phase in trial {self.name} due to missing stance phase indices.")
                    return None
                stance_phase_dfs = []
                for leg, indices in stance_phase_indices.items():
                    if indices['end_index'] == -1:
                        print(f"Cannot extract IK results for stance phase of {leg} in trial {self.name} due to missing end index.")
                        continue
                    start_idx, end_idx = indices['start_index'], indices['end_index']
                    stance_phase_df = self.IK_results.iloc[start_idx:end_idx].reset_index(drop=True)
                    stance_phase_df = time_normalize_df(stance_phase_df)
                    stance_phase_df['leg'] = leg
                    stance_phase_dfs.append(stance_phase_df)
                self._IK_results_stance_phase = pd.concat(stance_phase_dfs, ignore_index=True)
            except Exception as e:
                print(f"Error occurred while extracting IK results for stance phase in trial {self.name}: {e}")
                return None
        return self._IK_results_stance_phase

    def get_right_heel_strike_idx(self) -> Optional[int]:
        """Get the index of the right heel strike in the trial."""
        fp = deepcopy(self.c3d_handler.forces['forceplate_0']) #copy the forceplate data to avoid modifying the original data in the handler
        downsampling_ratio = fp.sampling_rate / self.c3d_handler.markers['R_ToesTop'].sampling_rate
        fp.downsample(int(downsampling_ratio))
        vertical_forces = fp.Fz
        threshold = 20 #heel strike threshold
        above_threshold = vertical_forces > threshold
        if not np.any(above_threshold):
            print(f"Warning: No right heel strike detected in trial {self.name}. Cannot determine right heel strike index.")
            return None
        heel_strike_1 = np.where(np.diff(above_threshold.astype(int)) == 1)[0][0] + 1
        leg_1 = 'R' if self.c3d_handler.markers['R_ToesTop'].x[heel_strike_1] > self.c3d_handler.markers['L_ToesTop'].x[heel_strike_1] else 'L'
        if leg_1 == 'R':
            return heel_strike_1
        else:
            fp2 = deepcopy(self.c3d_handler.forces['forceplate_1'])
            fp2.downsample(int(downsampling_ratio))
            vertical_forces_fp2 = fp2.Fz
            above_threshold_fp2 = vertical_forces_fp2 > threshold
            heel_strike_2 = np.where(np.diff(above_threshold_fp2.astype(int)) == 1)[0][0] + 1
            return heel_strike_2