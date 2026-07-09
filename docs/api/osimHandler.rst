OsimHandler
=========

.. automodule:: ibo_biomech.handlers.osimHandler
   :members:
   :show-inheritance:


.. code-block:: python


   #### EXAMPLE USAGE ####
   from ibo_biomech.osimHandler import OsimHandler
   #Setup files are still necessary. Since it is easier to use the OpenSim GUI to create tracking weights and markerSets.

   OsimHandler.run_ik(model_path=model, 
                   setup_file='example_IK_setup.xml', 
                   trc_file='test_data.trc', 
                   output_file='test_output/test_output_IK.mot', 
                   initial_time=0, 
                   final_time=1,
                   log_file='IK_logs.log')

   #Same anaylsis can be run directly with an h5 file without trc conversion
   OsimHandler.run_ik(model_path=model, 
                   setup_file='example_IK_setup.xml', 
                   h5_file='test_output.h5', 
                   output_file='test_output/test_output_IK.mot', 
                   initial_time=0, 
                   final_time=1,
                   log_file='IK_logs.log')
   
   #Scaling also supports both trc and h5 files. 
   #Any of the optional parameters (mass, height etc) will be pulled from the setup file if they are not provided as an argument. 
   OsimHandler.run_scaling(model_path=model, 
                        setup_file='example_scale_setup.xml', 
                        move_markers = True, 
                        h5_file=h5_file, 
                        output_file='test_output/test_output_scaled.osim', 
                        mass=70,
                        initial_time=0.5,
                        final_time=0.6)

   OsimHandler.run_id(model_path=model,
                   setup_file=setup_file,
                   mot_file='test_output/test_output_IK.mot',
                   external_loads_file=external_forces_file,
                   output_file='test_output/test_output_id.sto',
                   initial_time=0,
                   final_time=1)

   #If you don't have an external forces file, there is a utility for easier creation
   from ibo_biomech.utils import utils
   ext_loads = utils.build_extloads(r_idx=1,  #Which forceplate to use for right leg. left leg is automatically assigned.
                                    output_file='test_output/test_output_external_loads.xml', 
                                    h5_file=h5_file)
   #Which returns the output file path so you can just use it in the run id directly
   OsimHandler.run_id(model_path=model,
                   setup_file=setup_file,
                   mot_file='test_output/test_output_IK.mot',
                   external_loads_file=ext_loads,
                   output_file='test_output/test_output_id.sto',
                   initial_time=0,
                   final_time=1)