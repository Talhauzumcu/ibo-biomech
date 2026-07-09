"""OpenSim tooling wrappers.

This module defines :class:`OsimHandler`, wrappers around the OpenSim
Python API for running Inverse Kinematics, Scaling and Inverse Dynamics.

"""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import opensim as osim
import os
from ibo_biomech.utils.utils import read_storage

class OsimHandler:
    """Run OpenSim tools such as Inverse Kinematics and Scaling."""

    @staticmethod
    def run_ik(model_path: str=None, 
               setup_file: str=None,
               h5_file: str = None, 
               output_file: str=None,
               initial_time: float=None, 
               final_time: float=None,
               trc_file: str=None,
               log_file: str=None
               ) -> None:
        """Run Inverse Kinematics with OpenSim. A model and a setup file are 
        required to run ik. The function also accepts h5_file, which has 
        the marker information. Alternatively a trc_file can be directly provided. 
        If output, initial time or final time are provided, they will be used 
        instead of the setup file values. 

        Args:
            model_path: Path to the ``.osim`` model.
            h5_file: Path to the HDF5 file containing trial data.
            output_file: Output motion file path for the IK results.
            setup_file: Path to the IK setup file.
            initial_time: Start time in seconds; defaults to the TRC start time.
            final_time: End time in seconds; defaults to the TRC end time.
            log_file: Path to the log file for the IK process.
        """

        if h5_file is not None:
            trc_file = OsimHandler._trc_from_h5(h5_file) #Needs to be removed after use

        if model_path is None:
            raise ValueError("a valid model_path must be provided.")

        if trc_file is None and h5_file is None:
            raise ValueError("Either a trc_file or h5_file must be provided.")
        
        if setup_file is None:
            raise ValueError("a valid setup_file must be provided.")

        model = osim.Model(str(model_path))
        model.initSystem()
        setup_file = Path(setup_file).resolve()

        ikTool = osim.InverseKinematicsTool(str(setup_file))
        
        #Get the variables from setup if not provided to the function
        output_file = output_file if output_file else ikTool.getOutputMotionFileName()
        log_file = log_file if log_file else output_file.replace('.mot', '.log')
        trc_file = trc_file if trc_file else ikTool.getMarkerDataFileName()
        markerData = osim.MarkerData(str(trc_file))
        if initial_time is None:
            initial_time = markerData.getStartFrameTime()
        if final_time is None:
            final_time = markerData.getLastFrameTime()
        
        if not os.path.exists(output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists   

        # Setup logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink(str(log_file))
        osim.Logger.setLevelString("Info")

        print(f"Running IK with model: {model_path}, TRC: {trc_file}")
        ikTool.setModel(model)
        ikTool.setStartTime(initial_time)
        ikTool.setEndTime(final_time)
        ikTool.setMarkerDataFileName(str(trc_file))
        ikTool.setOutputMotionFileName(str(output_file))
        ikTool.set_report_errors(True)
        ikTool.set_report_marker_locations(True)
        ikTool.run()

        if log_file:
            osim.Logger.removeFileSink()  # Clean up logger to prevent issues with subsequent runs
        if h5_file is not None:
            os.remove(trc_file)  # Clean up the temporary TRC file if it was created from HDF5

    @staticmethod
    def run_scaling(model_path: str, 
                    setup_file: str,
                    trc_file: str,
                    h5_file: str=None,
                    mass: float=None,
                    height: float=None,
                    initial_time: float=None,
                    final_time: float=None,
                    move_markers: bool=None,
                    output_file: str=None,
                    log_file: str=None
                    ) -> str:
        """Scale an OpenSim model to a static trial. If provided optional parameters
        override values from setup.xml

        Args:
            model_path: Path to the generic ``.osim`` model.
            trc_file: Path to the static-pose marker TRC file.
            h5_file: Path to the HDF5 file containing marker data.
            mass: Subject mass in kilograms used to scale segment masses; if
                ``None`` the value from the setup file is kept.
            height: Subject height in meters used to scale segment lengths; if
                ``None`` the value from the setup file is kept.
            initial_time: Start time in seconds; defaults to the TRC start time.
            final_time: End time in seconds; defaults to the TRC end time.
            move_markers: If True, markers will be moved to match the scaled model. Fixed markers on the model will not be moved even if 
            the move markers is set to True.
            output_file: Output path for the scaled model.
            log_file: Path to the log file for the scaling process.

        Returns:
            Path to the scaled model file.
        """
        if h5_file is not None:
            trc_file = OsimHandler._trc_from_h5(h5_file) #Needs to be removed after use

        if model_path is None:
            raise ValueError("a valid model_path must be provided.")

        if trc_file is None and h5_file is None:
            raise ValueError("Either a trc_file or h5_file must be provided.")
        
        if setup_file is None:
            raise ValueError("a valid setup_file must be provided.")

        print(f"Running Scaling with model: {model_path}, TRC: {trc_file}")
        setup_file = Path(setup_file).resolve()
        scalingTool = osim.ScaleTool(str(setup_file))
        output_file = output_file if output_file is not None else scalingTool.getModelScaler().getOutputModelFileName()
        log_file = log_file if log_file is not None else output_file.replace('.osim', '.log')
        mass = mass if mass is not None else scalingTool.getSubjectMass()
        height = height if height is not None else scalingTool.getSubjectHeight()
        markerData = osim.MarkerData(str(trc_file))
        move_markers = move_markers if move_markers is not None else scalingTool.getMarkerPlacer().getApply()
        initial_time = initial_time if initial_time is not None else markerData.getStartFrameTime()
        final_time = final_time if final_time is not None else markerData.getLastFrameTime()

        if not os.path.exists(output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists   

        #Setup Logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink(str(log_file))
        osim.Logger.setLevelString("Info")

        # #Need this because opensim is stupid.
        setup_dir = setup_file.parent
        rel_model_path = os.path.relpath(model_path, setup_dir)
        rel_trc_path = os.path.relpath(trc_file, setup_dir)
        rel_output_path = os.path.relpath(output_file, setup_dir)
                
        time_range = osim.ArrayDouble()
        time_range.append(initial_time)
        time_range.append(final_time)
        
        scalingTool.getGenericModelMaker().setModelFileName(rel_model_path)

        modelScaler = scalingTool.getModelScaler()
        modelScaler.setMarkerFileName(rel_trc_path)
        modelScaler.setTimeRange(time_range)
        modelScaler.setOutputModelFileName(rel_output_path) 

        markerPlacer = scalingTool.getMarkerPlacer()
        markerPlacer.setApply(move_markers)
        markerPlacer.setStaticPoseFileName(rel_trc_path)
        markerPlacer.setTimeRange(time_range)
        markerPlacer.setOutputModelFileName(rel_output_path) 

        scalingTool.run()

        osim.Logger.removeFileSink()  # Clean up logger to prevent issues with subsequent runs
        return output_file

    @staticmethod
    def run_id(model_path: str, 
               setup_file: str,
               mot_file: str,
               external_loads_file: str=None,
               output_file: str=None,
               initial_time: float=None,
               final_time: float=None,
               lowpass_cutoff: float=-1.0,
               log_file: str=None
               ) -> str:
        """Run Inverse Dynamics with OpenSim. A model and a setup file are required to run id. 
        If output, initial time or final time are provided, they will be used 
        instead of the setup file values. lowpass_cutoff is used to filter the IK results before running ID.

        Args:
            model_path: Path to the ``.osim`` model.
            setup_file: Path to the ID setup file.
            mot_file: Path to the MOT file containing motion data (IK results).
            external_loads_file: Path to the external loads XML file;
            h5_file: Path to the HDF5 file containing trial data.
            output_file: Output motion file path for the ID results.
            initial_time: Start time in seconds; defaults to the TRC start time.
            final_time: End time in seconds; defaults to the TRC end time.
            lowpass_cutoff: Low-pass cutoff frequency for IK filtering; defaults to -1.0 (no filtering).
            log_file: Path to the log file for the ID process.
        Returns:
            Path to the output ID results file.
        """

        if model_path is None:
            raise ValueError("a valid model_path must be provided.")

        if setup_file is None:
            raise ValueError("a valid setup_file must be provided.")

        if external_loads_file is None:
            raise ValueError("a valid external_loads_file must be provided.")
        
        model = osim.Model(str(model_path))
        model.initSystem()
        setup_file = Path(setup_file).resolve()

        idTool = osim.InverseDynamicsTool(str(setup_file))
        
        #Get the variables from setup if not provided to the function
        output_file = output_file if output_file else idTool.getOutputGenForceFileName()
        log_file = log_file if log_file else output_file.replace('.sto', '.log')
        mot_file = mot_file if mot_file else idTool.getCoordinatesFileName()
        external_loads_file = external_loads_file if external_loads_file else idTool.getExternalLoadsFileName()
        
        if initial_time is None or final_time is None:
            d = read_storage(mot_file)
        initial_time = initial_time if initial_time is not None else float(d["time"][0])
        final_time = final_time if final_time is not None else float(d["time"][-1])

        if not os.path.exists(output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists   

        osim.Logger.removeFileSink()
        osim.Logger.addFileSink(str(log_file))
        osim.Logger.setLevelString("Info")

        idTool.setModel(model)
        idTool.setStartTime(initial_time)
        idTool.setEndTime(final_time)
        idTool.setCoordinatesFileName(str(mot_file))
        idTool.setExternalLoadsFileName(str(external_loads_file))
        idTool.setOutputGenForceFileName(str(Path(output_file).name))
        idTool.setResultsDir(str(Path(output_file).parent))
        idTool.setLowpassCutoffFrequency(lowpass_cutoff)
        excl = osim.ArrayStr(); excl.append("Muscles")
        idTool.setExcludedForces(excl)
        idTool.run()

        osim.Logger.removeFileSink()  # Clean up logger to prevent issues with subsequent runs
        return output_file
    
    @staticmethod
    def _trc_from_h5(h5_file: str):
        """Convert an HDF5 file to a TRC file to use with OpenSim"""

        from ibo_biomech import FileConverter
        trc_file = './._temp_trc.trc'
        FileConverter.h5_to_trc(h5_file, trc_file)
        return trc_file
    
    @staticmethod
    def _mot_from_h5(h5_file: str):
        """Convert an HDF5 file to a MOT file to use with OpenSim"""

        from ibo_biomech import FileConverter
        mot_file = './._temp_mot.mot'
        FileConverter.h5_to_mot(h5_file, mot_file)
        return mot_file
    