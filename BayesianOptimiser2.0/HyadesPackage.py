import os
import subprocess
import time
import sys
import pandas as pd
import numpy as np

class Hyades:
    def __init__(self, base_input_deck, final_laser_cutoff_time, LaserPowerFormula, logger, GenerateYFromCDF, output_directory):
        """
        Initialize the Hyades class with essential parameters and methods.

        This constructor initializes the Hyades class, setting up essential attributes
        that will be used in the simulation process. It prepares the necessary configurations
        such as the base input deck, laser cutoff time, power formula, logger, and output directory.

        Parameters:
        - base_input_deck (str): Path to the base input deck file used in simulations.
        - final_laser_cutoff_time (float): The final cutoff time for the laser in the simulations.
        - LaserPowerFormula (function): A function to calculate the laser power based on input parameters.
        - logger (logging.Logger): Logger object for logging information during the simulation process.
        - GenerateYFromCDF (function): A function to generate Y values from the simulation's CDF file.
        - output_directory (str): Directory where the simulation outputs will be stored.
        """
        self.base_input_deck = base_input_deck
        self.final_laser_cutoff_time = final_laser_cutoff_time
        self.LaserPowerFormula = LaserPowerFormula
        self.logger = logger
        self.GenerateYFromCDF = GenerateYFromCDF
        self.output_directory = output_directory
    
    def CreateOuterOutputDirectory(self):
        """
        Create the outer output directory for the experiment.

        This method checks if the output directory already exists. If it does, 
        the experiment is halted. Otherwise, the directory is created.
        """
        # Check the output_directory has been made
        if os.path.isdir(self.output_directory):
            self.logger.info('The output_directory already exists. Quitting the experiment')
            print('The output_directory already exists. Quitting the experiment')
            sys.exit(1)  # Exit the program to avoid overwriting existing data
        
        self.logger.info('Creating the output directory.')
        print('Creating the output directory.')
        subprocess.run(['mkdir', self.output_directory], check=True, text=True, capture_output=True)

    def WriteSubfile(self, raw_X, verbose):
        """
        Write a batch submission script for all X values in the batch.

        This method generates a SLURM batch submission script that will execute the 
        simulations and convert the results for each array job. The script is written 
        to `self.subfile_path`, which is derived from the current iteration directory.

        Parameters:
        - raw_X (2D list or np.ndarray): The array of input parameter sets to be used 
        in the simulations. The size of this array determines the number of jobs in 
        the SLURM array.
        - verbose (bool): If True, prints the path of the submission file after it is written.
        """
        # Define the path to the SLURM submission script
        self.subfile_path = self.iteration_directory + '/submit.slurm'

        # Open the file at subfile_path in write mode
        with open(self.subfile_path, 'w') as f:
            # Write the SLURM job script header to the file
            f.write('#!/bin/bash')
            f.write('\n#SBATCH --ntasks=1')
            f.write('\n#SBATCH --time=01:45:00')
            f.write('\n#SBATCH --array=0-%i' %(len(raw_X) - 1))
            f.write('\n#SBATCH --wait')
            f.write('\n#SBATCH -o %s/BatchOutput.txt' %self.iteration_directory)
            
            # Change directory to the iteration directory
            f.write('\n\ncd %s' %self.iteration_directory)
            
            # Write the commands to execute the simulation and convert results for each array job
            # Repeat these as the command occasionally fails
            f.write('\n\nhyades S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt')
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            
            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %self.iteration_directory)

        # If verbose mode is enabled, print the path of the submission file
        if verbose: print('Written the submission file %s' %self.subfile_path)

    def CreateOutputDirectory(self, iteration_number):
        """
        Create a directory for the current iteration.

        This method generates a new directory for the current iteration inside the 
        output directory. The directory is named 'I<iteration_number>' where 
        <iteration_number> is the iteration number provided.

        Parameters:
        - iteration_number (int): The current iteration number used for naming the directory.
        """
        # Create a directory for the current iteration
        self.iteration_directory = self.output_directory + '/I%i' %iteration_number
        os.mkdir(self.iteration_directory)
        print('\n\nWritten output directory for iteration', iteration_number)
        self.logger.info(f'Created Output Directory: {self.iteration_directory}')

    def CreateSimulationDirectory(self, simulation_number):
        """
        Create a directory for a specific simulation within the current iteration.

        This method generates a new directory for a specific simulation inside the 
        iteration directory. The directory is named 'S<simulation_number>' where 
        <simulation_number> is the simulation number provided.

        Parameters:
        - simulation_number (int): The simulation number used for naming the directory.
        """
        self.simulation_directory = self.iteration_directory + '/S%i' %simulation_number
        os.mkdir(self.simulation_directory)

    def ReplaceLine(self, file_name, line_num, text):
        """
        Replace a specific line in a text file with new text.

        This method opens a text file, replaces the text on a specific line with 
        the provided text, and saves the changes. It's particularly useful for 
        modifying input decks.

        Parameters:
        - file_name (str): The path to the input deck or text file.
        - line_num (int): The line number to overwrite (0-indexed).
        - text (str): The new text to put on the specified line.
        """
        # Open and read the file
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Edit the specified line
        lines[line_num] = text + '\n'  # Ensure text ends with a newline

        # Write the edited file back to disk
        with open(file_name, 'w') as file:
            file.writelines(lines)

    def WriteInputDeck3Pulse(self, simulation_input_deck, raw_X, simulation_number):
        """
        Write the input deck for the 3-pulse simulation using the raw_X values.

        This method copies a base input deck and modifies it according to the provided 
        `raw_X` values for the specific simulation number. It adjusts laser timings 
        and power settings as defined by the `LaserPowerFormula`.

        Parameters:
        - simulation_input_deck (str): Path to write the new input deck file.
        - raw_X (1D array): The input parameters for the simulation.
        - simulation_number (int): The index of the simulation in the batch.
        """
        # Copy base file into simulation directory
        subprocess.run(['cp', self.base_input_deck, simulation_input_deck])

        # Calculate values to change in the input deck

        # Laser timings 
        laser3_start_time = raw_X[simulation_number, 1] * self.final_laser_cutoff_time
        laser2_start_time = raw_X[simulation_number, 0] * laser3_start_time
        laser1_start_time = 2.000e-10

        # Laser powers
        laser3_power = self.LaserPowerFormula(raw_X[simulation_number, 4])
        laser2_power = raw_X[simulation_number, 3] * laser3_power
        laser1_power = raw_X[simulation_number, 2] * laser2_power

        # Replace lines in the input deck with calculated laser timings and powers
        self.ReplaceLine(simulation_input_deck, 31, f'tv {laser1_start_time:.3e} {laser1_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 32, f'tv {laser2_start_time:.3e} {laser1_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 33, f'tv {(laser2_start_time + 0.2e-9):.3e} {laser2_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 34, f'tv {laser3_start_time:.3e} {laser2_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 35, f'tv {(laser3_start_time + 0.2e-9):.3e} {laser3_power:.3e}')

    def WriteInputDeck4Pulse(self, simulation_input_deck, raw_X, simulation_number):
        """
        Write the input deck for the 4-pulse simulation using the raw_X values.

        This method copies a base input deck and modifies it according to the provided 
        `raw_X` values for the specific simulation number. It adjusts laser timings 
        and power settings as defined by the `LaserPowerFormula`.

        Parameters:
        - simulation_input_deck (str): Path to write the new input deck file.
        - raw_X (1D array): The input parameters for the simulation.
        - simulation_number (int): The index of the simulation in the batch.
        """
        # Copy base file into simulation directory
        subprocess.run(['cp', self.base_input_deck, simulation_input_deck])

        # Calculate values to change in the input deck

        # Laser timings 
        laser4_start_time = raw_X[simulation_number, 2] * self.final_laser_cutoff_time
        laser3_start_time = raw_X[simulation_number, 1] * laser4_start_time
        laser2_start_time = raw_X[simulation_number, 0] * laser3_start_time
        laser1_start_time = 2.000e-10

        # Laser powers
        laser4_power = self.LaserPowerFormula(raw_X[simulation_number, 6])
        laser3_power = raw_X[simulation_number, 5] * laser4_power
        laser2_power = raw_X[simulation_number, 4] * laser3_power
        laser1_power = raw_X[simulation_number, 3] * laser2_power

        # Replace lines in the input deck with calculated laser timings and powers
        self.ReplaceLine(simulation_input_deck, 31, f'tv {laser1_start_time:.3e} {laser1_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 32, f'tv {laser2_start_time:.3e} {laser1_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 33, f'tv {(laser2_start_time + 0.2e-9):.3e} {laser2_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 34, f'tv {laser3_start_time:.3e} {laser2_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 35, f'tv {(laser3_start_time + 0.2e-9):.3e} {laser3_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 36, f'tv {laser4_start_time:.3e} {laser3_power:.3e}')
        self.ReplaceLine(simulation_input_deck, 37, f'tv {(laser4_start_time + 0.2e-9):.3e} {laser4_power:.3e}')

    def ProcessOutputs(self, raw_X, iteration_number):
        """
        Process the output of each simulation.

        This method reads the output files generated by each simulation, processes 
        them to extract relevant data, and compiles the results into an array.

        Parameters:
        - raw_X (2D array): The array of input parameters used for the simulations.
        - iteration_number (int): The current iteration number.

        Returns:
        - np.ndarray: A 1D array of output values corresponding to the input parameters.
        """
        raw_y = []  # Initialize a list to store the output values

        # Process the output of each simulation
        for simulation_number, X in enumerate(raw_X):
            simulation_CDF_file = self.iteration_directory + '/S%i' %simulation_number + '/input%i.cdf' %simulation_number
            raw_y.append(self.GenerateYFromCDF(simulation_CDF_file, self.logger, iteration_number, simulation_number))

        raw_y = np.array(raw_y)   # Convert the list of Y values to a numpy array

        self.logger.info(f'raw_y has been made into an array. It is now {raw_y}')
        self.logger.info('')

        return raw_y

    def RunHyades3Pulse(self, raw_X, iteration_number, verbose):
        """
        Run the Hyades simulation for a given batch of X values for a 3-pulse laser profile.

        This method prepares the output directory, writes the batch submission file, 
        and creates the necessary simulation directories. It then submits the SLURM batch 
        submission script to run the simulations and waits for them to complete.

        Parameters:
        - raw_X (2D array): The batch of input parameters to run the simulation.
        - iteration_number (int): The current iteration number.
        - verbose (bool): If True, prints additional information during the run.
        
        Returns:
        - np.ndarray: The array of output results from the simulations.
        """
        # Create the output directory for the current iteration
        self.CreateOutputDirectory(iteration_number)

        # Write the SLURM batch submission file
        self.WriteSubfile(raw_X, verbose)

        # Create directories for each simulation and write the input decks
        for simulation_number, X in enumerate(raw_X):
            # Create a directory for the current simulation
            self.CreateSimulationDirectory(simulation_number)

            # Define the path to the simulation's input deck
            simulation_input_deck = self.simulation_directory + '/input%i.inf' %simulation_number
            
            # Write the input deck for the current simulation
            self.WriteInputDeck3Pulse(simulation_input_deck, raw_X, simulation_number)

        self.logger.info('Starting Hyades run')
        Hyades_start_time = time.time()  # Record the start time of the Hyades run


        # Submit the SLURM batch submission script and wait for jobs to complete
        print('Submitting the slurm submission file and waiting for completion')
        subprocess.run(['sbatch', self.subfile_path])
        print('Jobs completed.')

        Hyades_end_time = time.time()  # Record the end time of the Hyades run
        self.logger.info(f'Hyades has finished running and took {(Hyades_end_time-Hyades_start_time)/60} minutes.')

        # Process the outputs from the simulations into a 1D numpy array
        raw_y = self.ProcessOutputs(raw_X, iteration_number)

        return raw_y
    
    def RunHyades4Pulse(self, raw_X, iteration_number, verbose):
        """
        Run the Hyades simulation for a given batch of X values for a 4-pulse laser profile.

        This method prepares the output directory, writes the batch submission file, 
        and creates the necessary simulation directories. It then submits the SLURM batch 
        submission script to run the simulations and waits for them to complete.

        Parameters:
        - raw_X (2D array): The batch of input parameters to run the simulation.
        - iteration_number (int): The current iteration number.
        - verbose (bool): If True, prints additional information during the run.
        
        Returns:
        - np.ndarray: The array of output results from the simulations.
        """
        # Create the output directory for the current iteration
        self.CreateOutputDirectory(iteration_number)

        # Write the SLURM batch submission file
        self.WriteSubfile(raw_X, verbose)

        # Create directories for each simulation and write the input decks
        for simulation_number, X in enumerate(raw_X):
            # Create a directory for the current simulation
            self.CreateSimulationDirectory(simulation_number)

            # Define the path to the simulation's input deck
            simulation_input_deck = self.simulation_directory + '/input%i.inf' %simulation_number
            
            # Write the input deck for the current simulation
            self.WriteInputDeck4Pulse(simulation_input_deck, raw_X, simulation_number)

        self.logger.info('Starting Hyades run')
        Hyades_start_time = time.time()  # Record the start time of the Hyades run


        # Submit the SLURM batch submission script and wait for jobs to complete
        print('Submitting the slurm submission file and waiting for completion')
        subprocess.run(['sbatch', self.subfile_path])
        print('Jobs completed.')

        Hyades_end_time = time.time()  # Record the end time of the Hyades run
        self.logger.info(f'Hyades has finished running and took {(Hyades_end_time-Hyades_start_time)/60} minutes.')

        # Process the outputs from the simulations into a 1D numpy array
        raw_y = self.ProcessOutputs(raw_X, iteration_number)

        return raw_y

    def DeleteFiles(self, raw_y, iteration_number, stuck_in_peak_flag, batch_size):   
        """
        Delete files from the previous iteration if the optimizer is stuck in a peak.

        This method deletes the directories of simulations that did not produce the best result 
        from the previous iteration. If the optimizer is stuck in a peak, it deletes the entire 
        directory for the previous iteration.

        Parameters:
        - raw_y (1D array): The output results from the current iteration.
        - iteration_number (int): The current iteration number.
        """
        # Check if the optimizer is stuck in a peak
        if stuck_in_peak_flag == 0:
            # Find the index of the best result from the current iteration
            best_result_index = np.argmax(raw_y)
            self.logger.info(f'The most recent iteration (iteration {iteration_number-1}) produced a better result than previous iterations')
            
            # Iterate over each simulation in the batch
            for i in range(batch_size):
                if i != best_result_index:
                    try:
                        # Remove the sub-directory corresponding to simulations that were not the best
                        subprocess.run(['rm', '-rf', self.output_directory + '/I' + f'{iteration_number-1}' + '/S' + f'{i}'], check=True, text=True, capture_output=True)
                        self.logger.info(f"Directory removed successfully.")
                    except subprocess.CalledProcessError as e:
                        self.logger.info(f"Error: {e.stderr}")
                    except FileNotFoundError:
                        self.logger.info(f"Error: The directory does not exist.")
                    except PermissionError:
                        self.logger.info(f"Error: You do not have permission to delete.")
            
            # Iterate over all previous iterations to remove corresponding directories
            for i in range(iteration_number - 1):
                try:
                    subprocess.run(['rm', '-rf', self.output_directory + '/I' + f'{i}'], check=True, text=True, capture_output=True)
                    self.logger.info(f"Directory removed successfully.")
                except subprocess.CalledProcessError as e:
                    self.logger.info(f"Error: {e.stderr}")
                except FileNotFoundError:
                    self.logger.info(f"Error: The directory does not exist.")
                except PermissionError:
                    self.logger.info(f"Error: You do not have permission to delete.")
                
        else: 
            self.logger.info(f'The most recent iteration (iteration {iteration_number-1}) did not produce a better result than previous iterations')
            try:
                # Remove the entire directory for the previous iteration if stuck in peak
                subprocess.run(['rm', '-rf', self.output_directory + '/I' + f'{iteration_number-1}'], check=True, text=True, capture_output=True)
                self.logger.info(f"Directory removed successfully.")
            except subprocess.CalledProcessError as e:
                self.logger.info(f"Error: {e.stderr}")
            except FileNotFoundError:
                self.logger.info(f"Error: The directory does not exist.")
            except PermissionError:
                self.logger.info(f"Error: You do not have permission to delete.")    