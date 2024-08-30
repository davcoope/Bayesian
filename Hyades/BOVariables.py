import numpy as np

# # BO_Outputs_5 – using multipliers
# # 0.5 size capsule, 4 pulse

# BO_output_directory_number = 5

# batch_size = 200
# sub_batch_size = 50
# random_batch_size = 500

# iterations = 98
# random_iterations = 2

# acquisition_samples = 20000
# max_kappa = 4
# min_kappa = 0.1
# length_scale = 0.1

# reduce_bounds_factor = 0.5
# first_reduce_bounds = 15
# iterations_between_reducing_bounds = 3

# random_seed = 10

# base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'

# bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])

# log_path = f'/work4/clf/david/Bayesian/OutputsBO/log_{BO_output_directory_number}.log'
# csv_path = f'/work4/clf/david/Bayesian/OutputsBO/Outputs_{BO_output_directory_number}/Results.csv'
# output_directory = f'/work4/clf/david/Bayesian/OutputsBO/Outputs_{BO_output_directory_number}'

# final_laser_cutoff_time = 18e-9
# def LaserPowerFormula(x):
#     power = 2 * 10**(18 + (x * 3))
#     return(power)

# import numpy as np

# Test

BO_output_directory_number = 5

batch_size = 100
sub_batch_size = 50
random_batch_size = 500

iterations = 98
random_iterations = 2

acquisition_samples = 20000
max_kappa = 4
min_kappa = 0.1
length_scale = 0.1

reduce_bounds_factor = 0.5
first_reduce_bounds = 15
iterations_between_reducing_bounds = 3

random_seed = 10

base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'

bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1]])

log_path = f'/work4/clf/david/Bayesian/OutputsBO/log_{BO_output_directory_number}.log'
csv_path = f'/work4/clf/david/Bayesian/OutputsBO/Outputs_{BO_output_directory_number}/Results.csv'
output_directory = f'/work4/clf/david/Bayesian/OutputsBO/Outputs_{BO_output_directory_number}'

final_laser_cutoff_time = 18e-9
def LaserPowerFormula(x):
    power = 2 * 10**(18 + (x * 3))
    return(power)



