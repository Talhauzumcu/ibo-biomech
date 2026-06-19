"""OpenSim tooling wrappers.

This module defines :class:`OsimHandler`, thin wrappers around the OpenSim
Python API for running Inverse Kinematics and model Scaling from setup files.

Note:
    These helpers require the optional ``opensim`` and ``pandas`` packages and
    expect setup XML files under ``./setup_files/``.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import opensim as osim
import pandas as pd
import os
import numpy as np

class OsimHandler:
    """Run OpenSim tools such as Inverse Kinematics and Scaling."""

    @staticmethod
    def runIK(model_path: str, trc_path: str, mot_path: str, output: str, initial_time: float=None, final_time: float=None) -> None:
        """Run Inverse Kinematics with OpenSim.

        Args:
            model_path: Path to the ``.osim`` model.
            trc_path: Path to the marker TRC file.
            mot_path: Path to the motion/forces MOT file.
            output: Output motion file path for the IK results.
            initial_time: Start time in seconds; defaults to the TRC start time.
            final_time: End time in seconds; defaults to the TRC end time.
        """

        if not os.path.exists(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)

        print(f"/n/n/nOutput path for IK results: {output}/n/n/n")
        # Setup logger
        log_file = dir / (f'{output}_IK.log')
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink(str(log_file))
        osim.Logger.setLevelString("Info")

        print(f"Running IK with model: {model_path}, TRC: {trc_path}, MOT: {mot_path}")
        setup_file = Path('./setup_files/IKsetup.xml')
        markerData = osim.MarkerData(str(trc_path))
        if initial_time is None:
            initial_time = markerData.getStartFrameTime()
        if final_time is None:
            final_time = markerData.getLastFrameTime()
        model = osim.Model(str(model_path))
        model.initSystem()
        
        ikTool = osim.InverseKinematicsTool(str(setup_file))
        ikTool.setModel(model)
        ikTool.setStartTime(initial_time)
        ikTool.setEndTime(final_time)
        ikTool.setMarkerDataFileName(str(trc_path))
        ikTool.setOutputMotionFileName(str(output))
        ikTool.set_report_errors(True)
        ikTool.set_report_marker_locations(True)
        ikTool.run()

        osim.Logger.removeFileSink()  # Clean up logger to prevent issues with subsequent runs

    @staticmethod
    def runScaling(model_path: str, trc_path: str, output: str=None, mass: float=None) -> None:
        """Scale an OpenSim model to a static trial.

        Args:
            model_path: Path to the generic ``.osim`` model.
            trc_path: Path to the static-pose marker TRC file.
            output: Output path for the scaled model. Defaults to
                ``<model>_scaled.osim``.
            mass: Subject mass in kilograms used to scale segment masses; if
                ``None`` the value from the setup file is kept.
        """
        if output is None:
            output = os.path.splitext(model_path)[0] + "_scaled.osim"
        out_dir = os.path.dirname(output)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Running Scaling with model: {model_path}, TRC: {trc_path}")
        setup_file = Path('./setup_files/ScalingSetup.xml').resolve()
        
        #Need this because opensim is stupid.
        setup_dir = setup_file.parent
        rel_model_path = os.path.relpath(model_path, setup_dir)
        rel_trc_path = os.path.relpath(trc_path, setup_dir)
        rel_output_path = os.path.relpath(output, setup_dir)
        
        # 1. Get start and end times (Using the absolute path here is fine since MarkerData 
        # doesn't suffer from the XML document directory bug)
        markerData = osim.MarkerData(str(trc_path))
        initial_time = markerData.getStartFrameTime()
        final_time = markerData.getLastFrameTime()
        
        time_range = osim.ArrayDouble()
        time_range.append(initial_time)
        time_range.append(final_time)
        
        scalingTool = osim.ScaleTool(str(setup_file))
        scalingTool.getGenericModelMaker().setModelFileName(rel_model_path)
        if mass is not None:
            scalingTool.setSubjectMass(mass)

        modelScaler = scalingTool.getModelScaler()
        modelScaler.setMarkerFileName(rel_trc_path)
        modelScaler.setTimeRange(time_range)

        markerPlacer = scalingTool.getMarkerPlacer()
        markerPlacer.setStaticPoseFileName(rel_trc_path)
        markerPlacer.setTimeRange(time_range)
        markerPlacer.setOutputModelFileName(rel_output_path) 
        
        scalingTool.run()
