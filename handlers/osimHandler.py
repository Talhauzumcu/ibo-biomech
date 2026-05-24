from pathlib import Path
from typing import Any, Dict, Optional, Union
import opensim as osim
import pandas as pd
import os
import numpy as np

class OsimHandler:
    """Handler for running OpenSim tools like Inverse Kinematics and Scaling plus importing results into the data structure."""
    def __init__(self, trial_name: str = None, trial: Any = None):
        self.trial_name = trial_name
        self.trial = trial
        self._tables: Dict[str, osim.TimeSeriesTable] = {}
        self._dataframes: Dict[str, pd.DataFrame] = {}

    def attach_trial(self, trial: Any) -> None:
        """Attach a Trial instance so file paths can be inferred automatically."""
        self.trial = trial
        if getattr(trial, 'name', None) is not None:
            self.trial_name = trial.name

    def runIK(self, model_path: Path, trc_path: Path, mot_path: Path, output: Path=None, initial_time: float=None, final_time: float=None) -> None:
        """Run inverse kinematics using OpenSim."""
        if output is None:
            dir = Path(os.path.dirname(mot_path)) / 'IKResults'
            dir.mkdir(parents=True, exist_ok=True)
            output = dir / (f'{self.trial_name}_IK.mot') 
        print(f"/n/n/nOutput path for IK results: {output}/n/n/n")
        # Setup logger
        log_file = dir / (f'{self.trial_name}_IK.log')
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

    def runScaling(self, model_path: Path, trc_path: Path, output: Path=None, mass: float=None) -> None:
        """Run scaling using OpenSim."""
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

    def _resolve_default_result_path(self, result_type: str) -> Path:
        """Resolve default IK/ID output file path from the attached trial context."""
        if self.trial is None or not hasattr(self.trial, 'osimPath'):
            raise ValueError(
                "No Trial context attached. Provide a file path explicitly or attach a Trial instance first."
            )

        base_dir = Path(self.trial.osimPath).resolve()
        trial_name = self.trial.name

        if result_type == 'ik':
            candidate = base_dir / 'IKResults' / f'{trial_name}_IK.mot'
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f'No IK results found at expected location: {candidate}')

        if result_type == 'id':
            candidates = [
                base_dir / 'IDResults' / f'{trial_name}_ID.sto',
                base_dir / 'IDResults' / f'{trial_name}_ID.mot',
                base_dir / 'IDResults' / f'{trial_name}.sto',
                base_dir / 'IDResults' / f'{trial_name}.mot',
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                'No ID results found in expected locations. Checked: '
                + ', '.join(str(c) for c in candidates)
            )

        raise ValueError(f'Unsupported result_type: {result_type}')

    def read_table(self, file_path: Union[str, Path]) -> osim.TimeSeriesTable:
        """Read .sto/.mot via OpenSim's TableProcessor and return a TimeSeriesTable."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        table_processor = osim.TableProcessor(str(file_path))
        try:
            # Uses OpenSim's processing pipeline; with no extra operators this
            # still provides a robust file import path for .sto/.mot.
            table = table_processor.process()
        except Exception:
            table = osim.TimeSeriesTable(str(file_path))

        return table

    def table_to_dataframe(self, table: osim.TimeSeriesTable) -> pd.DataFrame:
        """Convert an OpenSim TimeSeriesTable to a pandas DataFrame."""
        num_rows = table.getNumRows()
        num_cols = table.getNumColumns()

        independent = table.getIndependentColumn()
        data: Dict[str, list] = {
            'time': np.array(independent)
        }

        for col_idx in range(num_cols):
            label = str(table.getColumnLabel(col_idx))
            dependent = table.getDependentColumn(label).to_numpy()
            data[label] = dependent

        return pd.DataFrame(data)

    def import_ik_results(
        self,
        file_path: Optional[Union[str, Path]] = None,
        as_dataframe: bool = False,
    ):
        """Import IK result file (.mot/.sto) using OpenSim tables."""
        try:
            resolved_path = Path(file_path).resolve() if file_path is not None else self._resolve_default_result_path('ik')
            table = self.read_table(resolved_path)
            self._tables['ik'] = table

            if as_dataframe:
                df = self.table_to_dataframe(table)
                self._dataframes['ik'] = df
                return df
            return table
        except Exception as e:
            print(f"Error importing IK results: {e}")
            return None

    def import_id_results(
        self,
        file_path: Optional[Union[str, Path]] = None,
        as_dataframe: bool = False,
    ):
        """Import ID result file (.mot/.sto) using OpenSim tables."""
        resolved_path = Path(file_path).resolve() if file_path is not None else self._resolve_default_result_path('id')
        table = self.read_table(resolved_path)
        self._tables['id'] = table

        if as_dataframe:
            df = self.table_to_dataframe(table)
            self._dataframes['id'] = df
            return df
        return table

    def get_ik_dataframe(self) -> pd.DataFrame:
        """Return IK results as a pandas DataFrame, loading them if needed."""
        if 'ik' in self._dataframes:
            return self._dataframes['ik']

        if 'ik' in self._tables:
            self._dataframes['ik'] = self.table_to_dataframe(self._tables['ik'])
            return self._dataframes['ik']

        return self.import_ik_results(as_dataframe=True)

    def get_id_dataframe(self) -> pd.DataFrame:
        """Return ID results as a pandas DataFrame, loading them if needed."""
        if 'id' in self._dataframes:
            return self._dataframes['id']

        if 'id' in self._tables:
            self._dataframes['id'] = self.table_to_dataframe(self._tables['id'])
            return self._dataframes['id']

        return self.import_id_results(as_dataframe=True)