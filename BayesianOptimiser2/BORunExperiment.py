import numpy as np

import BOPackage
import HyadesRuns.HyadesPackage as HyadesPackage
import HyadesRuns.BOVariables as Variables
import HyadesRuns.BOAnalysis as Analysis

def GenerateYFromCDF(simulation_CDF_file, logger, iteration_number, simulation_number):
    """
    Generate the Y value from the CDF file of a simulation.

    This method processes the CDF file from a simulation to extract key metrics 
    like gain, convergence ratio (CR), implosion factor (IFAR), velocity, and 
    parametric limits. It then applies various multipliers based on these metrics 
    to calculate a final Y value, which factors in the instability of the system.

    Parameters:
    - simulation_CDF_file (str): The path to the CDF file containing simulation data.
    - logger (logging.Logger): The logger object for logging messages.
    - iteration_number (int): The current iteration number for logging.
    - simulation_number (int): The current simulation number within the iteration for logging.

    Returns:
    - float: The final Y value after applying all relevant multipliers.
    """
    # If CDF file exists analyse.
    try:
        # Pull important values from the simulation
        gain_data = Analysis.Gain(simulation_CDF_file)
        CR_data = Analysis.ConvergenceRatio(simulation_CDF_file)
        IFAR_data = Analysis.IFAR(simulation_CDF_file)
        velocity_data = Analysis.ImplosionVelocity(simulation_CDF_file)
        
        # Extract gain value
        gain = gain_data[0]
        CR = np.max(CR_data[0])
        IFAR = IFAR_data[0][IFAR_data[1]]
        velocity = abs(min(velocity_data[0]))
        parametric_limit = Analysis.LaserProfile(simulation_CDF_file)[2]
        
        ############### Change multiplier back from 0, change velocity back to 400 and halve multiplier #################

        def GetMultiplier (x, X_cutoff, halve_width=False):
            """
            Calculate the multiplier based on the value and cutoff.

            This helper function computes a multiplier using a logistic function, 
            which is used to reduce the effective gain if certain conditions are met.

            Parameters:
            - X (float): The value of the metric (e.g., CR, IFAR).
            - X_cutoff (float): The cutoff value beyond which the multiplier reduces the gain.

            Returns:
            - float: The multiplier to be applied.
            """
            if halve_width:
                c = 2
            else:
                c = 1

            frac_x = (x - X_cutoff)/(X_cutoff)
            a = 0.9644
            b = 52
            return (1/a)*(1 - (1/(1 + np.exp(-(c*b*frac_x - 3.3)))))
        
        if CR > 16:
            CR_multiplier = GetMultiplier(CR, 16)
        else:
            CR_multiplier = 1

        if IFAR > 30:
            IFAR_multiplier = GetMultiplier(IFAR, 30)
        else:
            IFAR_multiplier = 1

        if velocity > 400:
            velocity_multiplier = GetMultiplier(velocity, 400, halve_width=True)
        else:
            velocity_multiplier = 1

        if parametric_limit > 1e14:
            parametric_limit_multiplier = GetMultiplier(parametric_limit, 1e14)
        else:
            parametric_limit_multiplier = 1

        # Combine all multipliers into a single value. This is where instability is factored into the loss function.
        result = gain * CR_multiplier * IFAR_multiplier * velocity_multiplier * parametric_limit_multiplier

    except:
        # If there is an issue processing the CDF file, return a default result of 0.0
        logger.info(f'There was an issue processing the CDF file for iteration {iteration_number}, simulation {simulation_number}. A result of 0.0 has been returned.')
        result = 0.0

    return [result]

bo = BOPackage.BO(
    KernelFunction=BOPackage.RBF_Kernel,
    length_scale=Variables.length_scale,
    AcquisitionFunction=BOPackage.UpperConfidenceBound,
    bounds=Variables.bounds,
    n_samples=Variables.n_samples,
    log_path=Variables.log_path,
    dynamic_bounds=True,
    iterations_between_reducing_bounds=Variables.iterations_between_reducing_bounds,
    first_reduce_bounds=Variables.first_reduce_bounds,
    reduce_bounds_factor=Variables.reduce_bounds_factor,
    random_seed=Variables.random_seed
    )

hyades = HyadesPackage.Hyades(
        base_input_deck=Variables.base_input_deck,
        final_laser_cutoff_time=Variables.final_laser_cutoff_time,
        LaserPowerFormula=Variables.LaserPowerFormula,
        logger=bo.logger,
        GenerateYFromCDF=GenerateYFromCDF,
        output_directory=Variables.output_directory
        )

# Create the output directory for all simulation results
hyades.CreateOuterOutputDirectory()

for i in range(Variables.random_iterations):
    raw_X = bo.GetRandomXBatch(Variables.random_batch_size)
    # Check the number of bounds to determine the type of laser profile
    if len(Variables.bounds) == 5:
        # If bounds length is 5, run the 3-pulse laser profile simulation
        raw_y = hyades.RunHyades3Pulse(raw_X, i, verbose=True)
    elif len(Variables.bounds) == 7:
        # If bounds length is 7, run the 4-pulse laser profile simulation
        raw_y = hyades.RunHyades4Pulse(raw_X, i, verbose=True)
    else:
        # Handle incorrect bounds length
        print("Incorrect bounds for a 3-pulse or 4-pulse laser profile.")

    # Write the simulation results to a CSV file
    bo.WriteOutputToCSV(Variables.csv_path, raw_X, raw_y)
    # Update the internal data storage with the new X and Y values
    bo.UpdateData(raw_X, raw_y)

    bo.LogCurrentStatus()

for i in range(Variables.iterations):
    raw_X = bo.GetNextXBatch(Variables.batch_size, sub_batch_size=Variables.sub_batch_size, max_kappa=Variables.max_kappa, min_kappa=Variables.min_kappa)
    # Check the number of bounds to determine the type of laser profile
    if len(Variables.bounds) == 5:
        # If bounds length is 5, run the 3-pulse laser profile simulation
        raw_y = hyades.RunHyades3Pulse(raw_X, i+Variables.random_iterations, verbose=True)
    elif len(Variables.bounds) == 7:
        # If bounds length is 7, run the 4-pulse laser profile simulation
        raw_y = hyades.RunHyades4Pulse(raw_X, i+Variables.random_iterations, verbose=True)
    else:
        # Handle incorrect bounds length
        print("Incorrect bounds for a 3-pulse or 4-pulse laser profile.")

    # Write the simulation results to a CSV file
    bo.WriteOutputToCSV(Variables.csv_path, raw_X, raw_y)
    # Update the internal data storage with the new X and Y values
    bo.UpdateData(raw_X, raw_y)

    bo.LogCurrentStatus()
