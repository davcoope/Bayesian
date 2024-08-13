import numpy as np

import BOPackage
import HyadesPackage
import Variables_BO as Variables
import Analysis

def KernelFunction(X1, X2, length_scale):
    """
    Radial Basis Function (RBF) kernel.

    Args:
        X1 (np.ndarray): First set of points.
        X2 (np.ndarray): Second set of points.
        length_scale (float): The length scale parameter.
        variance (float): The variance parameter.

    Returns:
        np.ndarray: The kernel matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / length_scale**2 * sqdist)

def AcquisitionFunction(mean, standard_deviation, kappa): 
    """
    Compute the acquisition value for a given set of parameters.

    This function calculates the acquisition value using the Upper Confidence Bound (UCB) method.
    The acquisition value is determined by combining the predicted mean, the standard deviation, 
    and a kappa value that balances exploration and exploitation.

    Parameters:
    - mean (float): The predicted mean value of the objective function.
    - standard_deviation (float): The standard deviation (uncertainty) of the prediction.
    - kappa (float): A parameter that controls the trade-off between exploration and exploitation.

    Returns:
    - float: The acquisition value, which is used to guide the selection of the next sample point.
    """
    return mean + kappa * standard_deviation

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
        
        def Get_multiplier (x, X_cutoff):
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
            frac_x = (x - X_cutoff)/(X_cutoff)
            a = 0.9644
            b = 52
            c = 3.3
            return (1/a)*(1 - (1/(1 + np.exp(-(b*frac_x - 3.3)))))
        
        # Reduce effective gain if CR is above 13. Reduces result by 1/e at a CR of 17
        if CR > 13:
            CR_multiplier = Get_multiplier(CR, 13)
        else:
            CR_multiplier = 1

        # Reduce effective gain if IFAR is above 30. Reduces result by 1/e at an IFAR of 40
        if IFAR > 30:
            IFAR_multiplier = Get_multiplier(IFAR, 30)
        else:
            IFAR_multiplier = 1

        # Reduce effective gain if IFAR is above 30. Reduces result by 1/e at an IFAR of 500
        if velocity > 400:
            velocity_multiplier = Get_multiplier(velocity, 400)
        else:
            velocity_multiplier = 1

        # Reduce effective gain if IFAR is above 1e14. Reduces result by 1/e at an parametric limit values of 2e14
        if parametric_limit > 1e14:
            parametric_limit_multiplier = Get_multiplier(parametric_limit, 1e14)
        else:
            parametric_limit_multiplier = 1

        # Combine all multipliers into a single value. This is where instability is factored into the loss function.
        result = gain * CR_multiplier * IFAR_multiplier * velocity_multiplier * parametric_limit_multiplier

    except:
        # If there is an issue processing the CDF file, return a default result of 0.0
        logger.info(f'There was an issue processing the CDF file for iteration {iteration_number}, simulation {simulation_number}. A result of 0.0 has been returned.')
        result = 0.0

    return [result]

def TestGenerateY(raw_X, logger):
    raw_y = []

    for i in range(len(raw_X)):
        y = 100 * (
            np.exp(-50 * ((raw_X[i,0] - 0.9)**2 + (raw_X[i,1] - 0.9)**2 + (raw_X[i,2] - 0.9)**2 + (raw_X[i,3] - 0.9)**2 + (raw_X[i,4] - 0.9)**2)) +
            5 * np.exp(-100 * ((raw_X[i,0] - 0.5)**2 + (raw_X[i,1] - 0.4)**2 + (raw_X[i,2] - 0.3)**2 + (raw_X[i,3] - 0.2)**2 + (raw_X[i,4] - 0.1)**2))
        )
        raw_y.append([y])

    raw_y = np.array(raw_y)

    logger.info(f'raw_y has been made into an array. It is now {raw_y}')
    logger.info('')

    return raw_y


bo = BOPackage.BO(
    log_path=Variables.log_path,
    KernelFunction=KernelFunction, 
    AcquisitionFunction=AcquisitionFunction, 
    bounds=Variables.bounds,
    n_samples=Variables.n_samples,
    length_scale=Variables.length_scale,
    iterations=Variables.iterations,
    batch_size=Variables.batch_size,
    max_kappa=Variables.max_kappa,
    min_kappa=Variables.min_kappa,
    output_directory=Variables.output_directory,
    random_seed=Variables.random_seed,
    iterations_between_reducing_bounds=Variables.iterations_between_reducing_bounds,
    first_reduce_bounds=Variables.first_reduce_bounds,
    reduce_bounds_factor=Variables.reduce_bounds_factor
    )

# Initialize the logger
bo.CreateLogger()

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

# Loop over the specified number of iterations
for j in range(Variables.iterations):
    if j == 0:
        # For the first iteration, generate random X values
        raw_X = bo.GetRandomXBatch()
    else:
        # For subsequent iterations, delete files from the previous iteration if necessary,
        # then generate new X values using the optimizer
        hyades.DeleteFiles(raw_y, j, bo.stuck_in_peak_flag, Variables.batch_size)
        raw_X = bo.GetNextXBatch(j)

    # Check the number of bounds to determine the type of laser profile
    if len(Variables.bounds) == 5:
        # If bounds length is 5, run the 3-pulse laser profile simulation
        raw_y = hyades.RunHyades3Pulse(raw_X, j, verbose=True)
    elif len(Variables.bounds) == 7:
        # If bounds length is 7, run the 4-pulse laser profile simulation
        raw_y = hyades.RunHyades4Pulse(raw_X, j, verbose=True)
    else:
        # Handle incorrect bounds length
        print("Incorrect bounds for a 3-pulse or 4-pulse laser profile.")

    # Write the simulation results to a CSV file
    bo.WriteOutputToCSV(raw_X, raw_y, j)
    # Update the internal data storage with the new X and Y values
    bo.UpdateData(raw_X, raw_y)
    # Check if the optimizer is stuck in a peak
    bo.StuckInPeak()
    # Reduce the bounds if necessary
    bo.ReduceBounds(j)
    
    bo.logger.info(f'Iteration {j} finished!!')

    # Log the current status of the optimization process
    bo.CurrentInfoStatus()

bo.logger.info('EXPERIMENT FINISHED!!')
bo.logger.info('')

# Log the final status of the optimization process
bo.TotalInfoStatus()