### think about changing np to cp / running matrix multiplication on GPU

import os
import time
import sys
import pandas as pd
import heapq
import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle

class BO:
    def __init__(self, KernelFunction, length_scale, AcquisitionFunction, bounds, n_samples, log_path=None, dynamic_bounds=False, iterations_between_reducing_bounds=None, first_reduce_bounds=None, reduce_bounds_factor=None, random_seed=52):
        """
        Initialize the Bayesian optimisation (BO) class with various parameters.
 
        Parameters:
        - KernelFunction (function): Function representing the kernel used in the Gaussian Process.
        - length_scale (float): The length scale of the kernel.
        - AcquisitionFunction (function): Function to calculate the acquisition value.
        - bounds (2D array): 2D array representing the bounds of each dimension.
        - n_samples (int): Number of samples to take. These are the candidate points. 

        Optional Parameters:

        Logging:
        - log_path (str): Path to the log file. No logging is done if log_path=None

        Dynamic Bounds:
        - dynamic_bounds (bool): Logical flag to use dynamic bounds
        - iterations_between_reducing_bounds (int): Only used if dynamic_bounds=True. Number of iterations without increasing maximum Y until the bounds are reduced.
        - first_reduce_bounds (int): Only used if dynamic_bounds=True. Minimum number of iterations before starting to reduce bounds.
        - reduce_bounds_factor (float): Only used if dynamic_bounds=True. Factor by which to reduce bounds.

        Random Seed:
        - random_seed (int): Seed for random number generation.
        """

        self.log_path = log_path

        self.Kernel = KernelFunction
        self.length_scale = length_scale

        self.AcquisitionFunction = AcquisitionFunction

        self.bounds = bounds
        self.n_samples = n_samples

        self.iteration_number = 0
        self.simulations_per_iteration = np.empty([0, 1])

        self.mean = None
        self.variance = None

        self.X_data = np.empty([0, 1])
        self.y_data = np.empty([0, 1])

        self.random_seed = random_seed

        if self.log_path is not None:
            self.CreateLogger()

        self.dynamic_bounds = dynamic_bounds
        if self.dynamic_bounds==True:
            self.iterations_between_reducing_bounds = iterations_between_reducing_bounds
            self.first_reduce_bounds = first_reduce_bounds
            self.reduce_bounds_factor = reduce_bounds_factor

            self.bounds_reduction_counter = 0
            self.stuck_in_peak_counter = 0


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - Get Next X - -- -- -- -------------------================ #

    def GetRandomXBatch(self, batch_size):
        """
        Generate a batch of random X values within the specified bounds.

        This method generates random points within the bounds for each batch of 
        simulations, ensuring that each point is unique within the batch.

        Returns:
        - raw_X (2D array): Randomly generated X values for the batch.
        """

        if self.log_path is not None:
            optimiser_start_time = time.time()  # Record start time for optimisation
            self.logger.info(f'Getting batch of random X values.')
            self.logger.info('')

        # Set the random seed for reproducibility
        np.random.seed(self.random_seed)

        raw_X = np.array([np.array([np.random.uniform(lower_bound, upper_bound) for (lower_bound, upper_bound) in self.bounds]) for i in range(batch_size)])


        if self.log_path is not None:
            optimiser_end_time = time.time()  # Record end time for optimisation
            self.logger.info(f'The time taken to get all X values for the random iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.')
            self.logger.info('')

        return raw_X
    
    def GetNextX(self, kappa=0.1, K_inv=None):
        """
        Get the next set of input parameters for the objective function.

        This method computes the next best set of input parameters by optimizing the
        acquisition function based on the predicted mean, standard deviation, and kappa value.

        Parameters:
        - current_simulation_number (int): The index of the current simulation within the batch.

        Returns:
        - np.ndarray: The next set of input parameters (X).
        """

        if self.X_data.size == 0:

            next_X = np.array([np.random.uniform(lower_bound, upper_bound) for (lower_bound, upper_bound) in self.bounds])
            
        else:

            if K_inv is None:
                K_inv = self.InverseKernel()

            # Generate a set of 'n_samples' candidate points from random samples within the bounds.
            candidate_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_samples, self.bounds.shape[0]))

            # Predict the mean and std at each of these points.
            self.mean, self.variance = self.PredictMeanVariance(candidate_x, K_inv=K_inv)

            # Draw samples from the posterior using the acquisition function
            candidate_y = self.AcquisitionFunction(self.mean, np.sqrt(self.variance), kappa)

            # Choose the x value which corresponds to the largest candidate y
            self.max_index = np.argmax(candidate_y)
            next_X = candidate_x[self.max_index]

        return next_X
    

    def GetNextXBatch(self, batch_size, kappa = 0.1, max_kappa = None, min_kappa = None, sub_batch_size=None):
        """
        Get the next batch of input parameters for the objective function.

        This method generates a batch of input parameters by iteratively optimizing 
        the acquisition function. It accounts for the exploration-exploitation trade-off 
        using kappa values and ensures uniqueness of the points.

        Parameters:
        - iteration_number (int): The current iteration number.

        Returns:
        - np.ndarray: The next batch of input parameters (X).
        """

        if self.X_data.size == 0:
            
            if self.log_path is not None:
                optimiser_start_time = time.time()  # Record start time for optimisation
                self.logger.info('Since there is no data stored in the object, this batch is random.')
                self.logger.info('Getting X values for this iteration')
                self.logger.info('')

            raw_X = self.GetRandomXBatch(batch_size)

            if self.log_path is not None:
                optimiser_end_time = time.time()  # Record end time for optimisation
                self.logger.info(f'The time taken to get all X values for this iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.')
                self.logger.info('')


        else:

            if self.log_path is not None:
                optimiser_start_time = time.time()  # Record start time for optimisation
                self.logger.info(f'Getting X values for this iteration')
                self.logger.info('')

            if sub_batch_size is None:

                raw_X = np.empty((batch_size,len(self.bounds)))  # Initialize the list to store the batch of X values
                
                K_inv = self.InverseKernel()

                for i in range(batch_size):
                    if max_kappa is not None and min_kappa is not None:
                        # Calculate kappa for the current simulation within the batch and use this in the acquisition function.
                        kappa = self.CalculateKappa(batch_size, i, max_kappa, min_kappa)

                    raw_X[i] = self.GetNextX(kappa, K_inv=K_inv)

            if sub_batch_size:

                raw_X = np.empty((batch_size,len(self.bounds)))  # Initialize the list to store the batch of X values

                for i in range(int(np.ceil(batch_size / sub_batch_size))):

                    sub_raw_X = np.empty((sub_batch_size, len(self.bounds)))
                    
                    sub_raw_y = np.empty([sub_batch_size, 1])

                    K_inv = self.InverseKernel()

                    for j in range(sub_batch_size):

                        if max_kappa is not None and min_kappa is not None:

                            kappa = self.CalculateKappa(sub_batch_size, j, max_kappa, min_kappa)

                        sub_raw_X[j] = self.GetNextX(kappa, K_inv=K_inv)

                        raw_X[j + sub_batch_size * i] = sub_raw_X[j]

                        normalised_y = self.mean[self.max_index]

                        sub_raw_y[j] = normalised_y * np.max(self.y_data) + np.min(self.y_data)

                    # Concatenate raw_X to the existing X_data array
                    self.X_data = np.vstack([self.X_data, sub_raw_X])

                    # Same for y
                    self.y_data = np.vstack([self.y_data, sub_raw_y])

                self.X_data = self.X_data[:-batch_size]
                self.y_data = self.y_data[:-batch_size]


        if self.log_path is not None:
            optimiser_end_time = time.time()  # Record end time for optimisation
            self.logger.info(f'The time taken to get all X values for this iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.')
            self.logger.info('')

        return raw_X

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- - Update object - -- -- -- -------------------================ #

    def UpdateData(self, raw_X, raw_y, update_iteration=True):
        """
        Update the internal data storage with new X and Y values.

        This method appends the new input parameters (X) and output results (Y) from 
        the current iteration to the existing data arrays stored in the class.

        Parameters:
        - raw_X (2D array): The new input parameters to append.
        - raw_y (1D array): The new output results to append.
        """

        # If X_data is empty, initialize it with raw_X. Otherwise, concatenate raw_X to the existing X_data array
        if self.X_data.size == 0:
            self.X_data = raw_X
        else:
            self.X_data = np.vstack([self.X_data, raw_X])

        # Same for y
        if self.y_data.size == 0:
            self.y_data = raw_y
        else:
            self.y_data = np.vstack([self.y_data, raw_y])

        # Update the iteration count
        if update_iteration:
            self.iteration_number += 1
            self.simulations_per_iteration = np.vstack([self.simulations_per_iteration, len(raw_y)])

        # Check for bounds reduction
        if self.dynamic_bounds==True:
            self.StuckInPeak()
            self.ReduceBounds()

        # Log current status
        if self.log_path is not None:
            self.LogCurrentStatus()

    def UpdateDataCSV(self, csv_file, update_iteration=False):
        """
        Update the internal data storage by reading values from a CSV file.

        This method reads a CSV file containing results from previous iterations 
        and updates the internal data storage (X_data and y_data) with the information.

        Parameters:
        - csv_file (str): The path to the CSV file containing previous results.
        """
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Extract the Y data from the DataFrame
        self.y_data = np.append(self.y_data, df['Result'].values)

        csv_data_length = len(df['Result'].values)

        # Initialize a zero array for the new X data with the correct shape
        self.X_data = np.vstack( [self.X_data, np.zeros((csv_data_length, len(self.bounds)))] )

        # Loop over each row of the newly added data and each column (each dimension of X)
        for i in range(csv_data_length):
            for k in range(len(self.bounds)):
                # Fill in the X data with the values from the DataFrame
                self.X_data[len(self.y_data) - csv_data_length +i][k] = df[f'X{k}'][i]

        # Update the iteration count
        if update_iteration:
            self.iteration_number += 1


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- - - Saving Data - -- -- -- -------------------================ #
    

    def WriteOutputToCSV(self, csv_path, raw_X, raw_y):
        """
        Write the simulation results to a CSV file.

        This method saves the results of the current iteration, including the input 
        parameters (X) and output results (Y), to a CSV file. If the file does not exist, 
        it creates a new one with headers. Otherwise, it appends the new data to the existing file.

        Parameters:
        - raw_X (2D array): The input parameters used in the current iteration.
        - raw_y (1D array): The output results corresponding to the input parameters.
        - iteration_number (int): The current iteration number, used to tag the data.
        """

        # Create arrays for iteration numbers and simulation numbers
        iteration_numbers = np.full(len(raw_X), self.iteration_number)
        simulation_numbers = range(0, len(raw_X))

        # Create a dictionary to hold the data with column names
        data = {
            'Iteration': np.array(iteration_numbers),
            'Simulation': np.array(simulation_numbers),
            'Result': raw_y[:, 0],
        }

        # Add raw_X values with column names (X0, X1, X2, ...)
        for i in range(np.shape(raw_X)[1]):
            data[f'X{i}'] = raw_X[:,i]

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data)

        # Check if the CSV file exists, if not, create it and write the headers
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            # Append new data to the existing CSV file
            df.to_csv(csv_path, mode='a', header=False, index=False)

        if self.log_path is not None:
            self.logger.info('csv file updated.')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - Algebra - - -- -- -- -------------------================ #


    def InverseKernel(self, jitter=1e-7):
        """
        Calculate the Kernel and inverse of the kernel matrices (K, K_inv) based on the recorded X data.

        This method adds a small jitter term to the kernel matrix for numerical stability,
        computes the inverse, and returns it.

        Returns:
        - np.ndarray: The inverse of the kernel matrix (K_inv).
        """

        # Compute the kernel matrix with the jitter term added
        K = self.Kernel(self.X_data, self.X_data, self.length_scale) + jitter * np.eye(len(self.X_data))

        if self.log_path is not None:
            inverting_start_time = time.time()  # Record start time for inversion

        
        # Calculate the inverse of the kernel matrix
        K_inv = np.linalg.inv(K)

        # Log the time taken to invert the kernel matrix
        if self.log_path is not None:
            inverting_end_time = time.time()
            self.logger.info(f'It took {inverting_end_time-inverting_start_time} to invert the kernel.')

        return K_inv

    def PredictMeanVariance(self, candidate_x, K_inv=None, jitter=1e-7):
        """
        Predict the mean and standard deviation of the objective function.

        This method uses the Gaussian Process (GP) to predict the mean and variance of
        the objective function at random points within the specified bounds.
        """

        if K_inv is None:
            K_inv = self.InverseKernel()

        K_star = self.Kernel(self.X_data, candidate_x, self.length_scale)
        K_star_star = self.Kernel(candidate_x, candidate_x, self.length_scale) + jitter

        # Shift and Normalize the observed data
        shifted_y_data = self.y_data - np.min(self.y_data)
        normalized_y_data = shifted_y_data / np.max(shifted_y_data)

        # Predict the mean of the new point
        mean = K_star.T.dot(K_inv).dot(normalized_y_data)  
        
        # Compute the full covariance matrix of the prediction
        full_cov = K_star_star - K_star.T.dot(K_inv).dot(K_star)

        # Extract the diagonal elements to get the variances for each new point
        var = np.diag(full_cov).reshape(len(candidate_x), 1)

        return mean, var
    
    def CalculateKappa(self, batch_size, current_simulation_number, max_kappa, min_kappa):
        """
        Compute the UCB parameter kappa for the current batch.

        Kappa determines the exploration-exploitation trade-off in Bayesian optimisation. 
        It is dynamically calculated based on the batch size and the current simulation number.

        Parameters:
        - current_simulation_number (int): The index of the current simulation within the batch.

        Returns:
        - float: The computed kappa value.
        """

        try:
            # Check if batch_size is equal to 1
            if batch_size == 1:
                # Compute kappa as the average of max_kappa and min_kappa
                kappa = (max_kappa - min_kappa) / 2
            else:
                # Calculate the exponential factor 'b'
                b = 1/(batch_size-1) * np.log(max_kappa / min_kappa)

                # Calculate kappa using an exponential function
                kappa = min_kappa * np.exp(b*current_simulation_number)

        # Handle any exceptions that occur during the try block
        except Exception as e:
            if self.log_path is not None:
                self.logger.info(f"An error occurred: {e}")   

        return kappa


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                                
    # ==============----------------- -- -- - Bounds Reduction - -- -- -------------------================ #   


    def StuckInPeak(self):
        """
        Check if the optimizer has become stuck at a peak.

        This method compares the best value from the most recent iteration with the 
        current best value. If no improvement is observed, it increases a counter 
        that tracks how long the optimizer has been stuck in a peak.
        """
        # Define the range of indices for the most recent iteration
        relevant_indices = range(max(len(self.y_data) - self.batch_size, 0), len(self.y_data))
        
        # Find the largest index from the most recent iteration
        largest_index = heapq.nlargest(1, relevant_indices, key=self.y_data.__getitem__)
        
        # Retrieve the actual Y values for these indices
        largest_value = [self.y_data[i] for i in largest_index]

        # Check if the largest value found is greater than the current best value
        if largest_value[0] > self.current_best_value:
            self.stuck_in_peak_flag = 0  # Not stuck in a peak
            self.stuck_in_peak_counter = 0  # Reset the counter
            self.current_best_value = largest_value[0]  # Update the current best value
        else:
            self.stuck_in_peak_flag = 1  # Stuck in a peak
            self.stuck_in_peak_counter += 1  # Increment the counter

            if self.log_path is not None:
                self.logger.info('The Optimiser has become stuck at a peak')  # Log the event
    
    def FindBounds(self, number, range):
        """
        Find the upper and lower bounds for the domain for a given number within a specified range.

        This method calculates the bounds for a given number, ensuring that the bounds 
        remain within the interval [0, 1]. This is particularly useful when dealing with 
        normalized data or probabilities.

        Parameters:
        - number (float): The central value for which to find bounds.
        - range (float): The range around the central value.

        Returns:
        - list: A list containing the lower and upper bounds.
        """
        # Calculate the upper bound by adding half of the range to the number
        upper_bound = number + range/2

        # Calculate the lower bound by subtracting half of the range from the number
        lower_bound = number - range/2

        # Check if the lower bound is less than 0
        if lower_bound < 0:
            lower_bound = 0   # If the lower bound is less than 0, set it to 0
            upper_bound = range  # Adjust the upper bound to be the full range from 0

        # Check if the upper bound exceeds 1    
        elif upper_bound > 1:
            lower_bound = 1-range  # Adjust the lower bound to be 1 minus the range
            upper_bound = 1   # If the upper bound exceeds 1, set it to 1

        return [lower_bound,upper_bound]
    
    def ReduceBounds(self):
        """
        Reduce the search bounds if stuck in a peak.

        This method reduces the search bounds if the optimizer has been stuck in a peak 
        for a specified number of iterations. It proportionally reduces the bounds and 
        the length scale used in the kernel function.

        Parameters:
        - iteration_number (int): The current iteration number.
        """

        if self.log_path is not None:
            self.logger.info(f'Stuck in peak counter is: {self.stuck_in_peak_counter}')

        # Reduce the search bounds if stuck in a peak and past the first_reduce_bounds threshold
        if self.stuck_in_peak_counter >= self.iterations_between_reducing_bounds and self.iteration_number >= self.first_reduce_bounds:
            self.stuck_in_peak_counter = 0  # Reset the stuck_in_peak_counter
            self.bounds_reduction_counter += 1  # Increment the bounds_reduction_counter
            spread = self.reduce_bounds_factor ** self.bounds_reduction_counter  # Set the correct range of the bounds
            self.length_scale = self.length_scale * self.reduce_bounds_factor  # Reduce the length scale proportionally to the bounds
            
            # Calculate the new bounds for this dimension using the current best X value and the spread
            bounds = np.empty([len(bounds), 2])
            for i in range(len(self.bounds)):
                bounds[i] = self.FindBounds(self.X_data[self.BestData()[0][0]][i], spread)
            self.bounds = np.array(bounds)


            if self.log_path is not None:
                self.logger.info(f'New bounds are {self.bounds}')

    def BestData(self):
        """
        Find the best data points in terms of the maximum observed values.

        This method identifies the largest observed values from the optimisation process,
        sorts them, and returns the sorted indices and values.

        Parameters:
        - y_data (list): The list of observed values.

        Returns:
        - tuple: A tuple containing the sorted indices and values.
        """
        # Set the number of indices to find
        number_indices = 1

        # Find the indices of the largest Y values
        largest_indices = heapq.nlargest(number_indices, range(len(self.y_data)), key=self.y_data.__getitem__)
        
        # Retrieve the Y values for these indices
        largest_values = np.array([self.y_data[i] for i in largest_indices])

        # Sort the indices and values into value order
        sorted_indices_and_values = sorted(zip(largest_indices, largest_values), key=lambda x: x[1], reverse=True)

        # Unzip the sorted pairs back into separate lists of indices and values
        sorted_indices, sorted_values = zip(*sorted_indices_and_values)

        return(sorted_indices, sorted_values)


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                                
    # ==============----------------- -- -- -- - - Logging -- -- -- -- -------------------================ #            

    def CreateLogger(self):
        """
        Create a logger for the optimisation process.

        This function checks if a log file already exists. If it does, the program exits
        to prevent overwriting. If not, a new logger is created and configured.

        Raises:
        - SystemExit: If the log file already exists.
        """

        # Check if the log file exists
        if os.path.exists(self.log_path):
            print(f"Error: The log file at {self.log_path} already exists. Quitting the experiment.")
            sys.exit(1)  # Exit to prevent overwriting the log file

        # Setup logger and set level to INFO
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Setup Log_handler - set mode to 'w' to write
        log_handler = logging.FileHandler(self.log_path, mode='w')
        log_handler.setLevel(logging.INFO)

        # Define the log format (preamble before your message is displayed)
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_format)

        # Add the handler to the logger object so you can start writing to the log file
        self.logger.addHandler(log_handler)

        self.logger.info('The log has been created')


    def LogCurrentStatus(self):
        """
        Log the current status of the optimisation process.

        This method logs the best value found so far, the corresponding X values, 
        the number of random X values used, and how many times the bounds have been reduced.
        """
        self.logger.info(f'Current best y value was {self.BestData()[1][0]}; the corresponding X values were {self.X_data[self.BestData()[0][0]]}')
        if self.dynamic_bounds is True:
            self.logger.info(f'The bounds have been reduced {self.bounds_reduction_counter} times')
        self.logger.info('')
        self.logger.info('')

# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- - - - Plotting - - - -- -- -------------------================ #   

def SausagePlot(object, highlight_recent=0, resolution=1000):
    plt.figure(figsize=(12, 6))
    
    if len(object.bounds) == 1:

        sample_points = np.linspace(0, 1, resolution, endpoint=True).reshape(resolution,1)

        mean, variance = object.PredictMeanVariance(sample_points)

        plt.plot(sample_points, mean, label='mean')
        plt.fill_between(sample_points[:,0], mean[:,0] - 1.96 * np.sqrt(variance[:,0]), mean[:,0] + 1.96 * np.sqrt(variance[:,0]), color = 'blue', alpha=0.2, label = '95% confidence interval')

        shifted_y_data = object.y_data - np.min(object.y_data)
        normalized_y_data = shifted_y_data / np.max(shifted_y_data)

        plt.scatter(object.X_data, normalized_y_data, s=10)
        
        if highlight_recent != 0:
            plt.scatter(object.X_data[-highlight_recent:], normalized_y_data[-highlight_recent:], s=30, color='red')

        plt.title("Mean/Varance Plot")

        plt.legend()

        # Display the plot
        plt.show()

    else:

        print('Can only produce sausage plots of one dimensional functions.')

def KappaAcquisitionFunctionPlot(object, number_kappas, number_candidate_points, max_kappa, min_kappa, resolution=1000):
    """
    Visualize the acquisition function for a range of kappa values.

    This function calculates and plots the acquisition function for a specified number of 
    kappa values. It generates candidate X points, calculates the corresponding Y values, 
    and identifies the maximum Y value for each kappa. The results are plotted to help 
    visualize the acquisition function and its dependency on the kappa value.

    Parameters:
    - object (object): The object that contains methods to calculate kappa, mean, and variance.
    - number_kappas (int): The number of kappa values to test.
    - number_candidate_points (int): The number of candidate X points to sample.
    - max_kappa (float): The maximum kappa value.
    - min_kappa (float): The minimum kappa value.
    - resolution (int): The number of points used to sample the acquisition function. Default is 1000.
    """

    plt.figure(figsize=(12, 6))

    # Function requires a one dimensional optimisation problem
    if len(object.bounds) == 1:

        kappas = np.empty([number_kappas, 1])  # Initialize an empty array to store kappa values
        
        # Calculate kappa values for the specified range
        for i in range(number_kappas):
            kappas[i] = object.CalculateKappa(number_kappas, i, max_kappa, min_kappa)

        # Generate evenly spaced sample points between the bounds
        sample_points = np.concatenate([np.linspace(lower_bound, upper_bound, resolution, endpoint=True) for (lower_bound, upper_bound) in object.bounds]).reshape(resolution,1)
        
        # Predict the mean and variance for the sample points
        mean, variance = object.PredictMeanVariance(sample_points)

        # Normalize the Y data for plotting
        shifted_y_data = object.y_data - np.min(object.y_data)
        normalized_y_data = shifted_y_data / np.max(shifted_y_data)

        # Plot the normalized data points on the graph
        plt.scatter(object.X_data, normalized_y_data, s=10)

        # Iterate through each kappa value to calculate the acquisition function
        for i in range(number_kappas):
            candidate_X = np.empty([number_candidate_points, len(object.bounds)])  # Initialize array to store candidate X points
            candidate_y = np.empty([number_candidate_points, 1])  # Initialize array to store corresponding y values
            
            # Generate candidate points and calculate their acquisition values
            for j in range(number_candidate_points):
                random_index = np.random.randint(0, resolution)  # Randomly select an index
                candidate_X[j] = sample_points[random_index]  # Store the corresponding X value
                candidate_y[j] = object.AcquisitionFunction(mean[random_index], np.sqrt(variance[random_index]), kappas[i])  # Calculate and store the acquisition value
                                               
            # Calculate the acquisition function for the entire sample space
            sample_y = object.AcquisitionFunction(mean, np.sqrt(variance), kappas[i])

            # Plot the acquisition function curve
            plt.plot(sample_points, sample_y, label=f'Kappa={round(kappas[i][0],2)}', color=f'C{i}')

            # Choose the x value which corresponds to the largest candidate y
            max_index = np.argmax(candidate_y)

            # Plot the point with the maximum acquisition value
            plt.scatter(candidate_X[max_index], candidate_y[max_index], s=50, color=f'C{i}')
            # Plot the other acquisition values
            plt.scatter(candidate_X, candidate_y, s=10,  color=f'C{i}')

        # Label the axes
        plt.xlabel('X')
        plt.ylabel('Acquisition Function Value')

        plt.title("Acquisition Function")

        # Display the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Display the plot
        plt.show()

    else:
        print('Function requires a one dimensional optimization problem.')


# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- -- - - Kernels - - -- -- -- -------------------================ #         


def RBF_Kernel(X1, X2, length_scale):
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


def MaternKernel(X1, X2, length_scale, nu=0.1):
    return 0



# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- Acquisition Functions - -- -------------------================ #    

def UCB(mean, standard_deviation, kappa):
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
    random_numbers = 0.01 * (np.random.rand(len(mean)).reshape(len(mean), 1))

    return mean + kappa * (standard_deviation + random_numbers)



# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- - Load/Save Object - -- -- -------------------================ #    



def SaveOptimisaer(object, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(object, file)



def LoadOptimiser(file_path):
    with open(file_path, 'rb') as file:
        object = pickle.load(file)

    return object