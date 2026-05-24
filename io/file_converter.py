import h5py
import numpy as np
from handlers.c3dHandler import C3DHandler

class FileConverter:
    """Stateless utility class to convert between biomechanical file formats."""
    
    @staticmethod
    def c3d_to_h5(c3d_path: str, h5_path: str) -> None:
        """Converts a C3D file directly to the lab's HDF5 architecture."""
        
        handler = C3DHandler(c3d_path) 
        handler.load_data()
        
        with h5py.File(h5_path, 'w') as h5f:
            
            if handler.markers:
                labels = list(handler.markers.keys())
                first_marker = handler.markers[labels[0]]
                num_frames = len(first_marker.x)
                
                traj_data = np.zeros((len(labels), 4, num_frames))
                for i, label in enumerate(labels):
                    traj_data[i, :, :] = handler.markers[label].get_frame_trajectory()
                
                traj_group = h5f.create_group("Trajectories")
                traj_group.create_dataset("Data", data=traj_data)
                traj_group.attrs["Labels"] = labels
                traj_group.attrs["SamplingFrequency"] = first_marker.sampling_rate
                traj_group.attrs["NumFrames"] = num_frames
                traj_group.attrs["NumLabeled"] = len(labels)

            # --- Force Plates ---
            if handler.forces:
                fp_group = h5f.create_group("ForcePlates")
                for i, (name, fp_data) in enumerate(handler.forces.items()):
                    plate_group = fp_group.create_group(str(i))
                    plate_group.attrs["Name"] = name
                    plate_group.attrs["SamplingFrequency"] = fp_data.sampling_rate
                    plate_group.attrs["NumSamples"] = fp_data.force.shape[0]
                    
                    # Transpose back to (3, n_samples) for H5 storage
                    plate_group.create_dataset("Force", data=fp_data.force.T)
                    plate_group.create_dataset("Moment", data=fp_data.moment.T)
                    plate_group.create_dataset("COP", data=fp_data.cop.T)
        
        # handler goes out of scope here and memory is safely freed
        print(f"Successfully converted {c3d_path} to {h5_path}")