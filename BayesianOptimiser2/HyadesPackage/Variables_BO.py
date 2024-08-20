import numpy as np

# # BO_Outputs_1
# # 0.5 size capsule, 3 pulse
# batch_size = 100
# iterations = 100
# n_samples = 10000
# max_kappa = 4
# min_kappa = 0.1
# length_scale = 0.1
# reduce_bounds_factor = 0.5
# first_reduce_bounds = 10
# iterations_between_reducing_bounds = 3
# bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])
# BO_output_directory_number = 1
# log_path = f'/work4/clf/david/Bayesian/Outputs/BO_log_{BO_output_directory_number}'
# output_directory = f'/work4/clf/david/Bayesian/Outputs/BO_Outputs_{BO_output_directory_number}'
# base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'
# random_seed = 10
# final_laser_cutoff_time = 18e-9
# def LaserPowerFormula(x):
#     power = 2 * 10**(18 + (x * 3))
#     return(power)

# # BO_Outputs_2
# # 0.5 size capsule, 4 pulse
# batch_size = 5
# iterations = 3
# n_samples = 1000
# max_kappa = 4
# min_kappa = 0.1
# length_scale = 0.1
# reduce_bounds_factor = 0.5
# first_reduce_bounds = 10
# iterations_between_reducing_bounds = 3
# bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])
# BO_output_directory_number = 2
# log_path = f'/work4/clf/david/Bayesian/Outputs/BO_log_{BO_output_directory_number}'
# output_directory = f'/work4/clf/david/Bayesian/Outputs/BO_Outputs_{BO_output_directory_number}'
# base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'
# random_seed = 10
# final_laser_cutoff_time = 18e-9
# def LaserPowerFormula(x):
#     power = 2 * 10**(18 + (x * 3))
#     return(power)

# # BO_Outputs_3
# # 0.5 size capsule, 4 pulse
# batch_size = 100
# iterations = 100
# n_samples = 10000
# max_kappa = 4
# min_kappa = 0.1
# length_scale = 0.1
# reduce_bounds_factor = 0.5
# first_reduce_bounds = 15
# iterations_between_reducing_bounds = 3
# bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])
# BO_output_directory_number = 3
# log_path = f'/work4/clf/david/Bayesian/Outputs/BO_log_{BO_output_directory_number}'
# output_directory = f'/work4/clf/david/Bayesian/Outputs/BO_Outputs_{BO_output_directory_number}'
# base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'
# random_seed = 10
# final_laser_cutoff_time = 18e-9
# def LaserPowerFormula(x):
#     power = 2 * 10**(18 + (x * 3))
#     return(power)

# BO_Outputs_4
# 0.5 size capsule, 4 pulse
batch_size = 3
iterations = 3
n_samples = 1000
max_kappa = 4
min_kappa = 0.1
length_scale = 0.1
reduce_bounds_factor = 0.5
first_reduce_bounds = 2
iterations_between_reducing_bounds = 3
bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])
BO_output_directory_number = 4
log_path = f'/work4/clf/david/Bayesian/Outputs/BO_log_{BO_output_directory_number}'
output_directory = f'/work4/clf/david/Bayesian/Outputs/BO_Outputs_{BO_output_directory_number}'
base_input_deck = '/work4/clf/david/Bayesian/InputDecks/infForDavid.inf'
random_seed = 10
final_laser_cutoff_time = 18e-9
def LaserPowerFormula(x):
    power = 2 * 10**(18 + (x * 3))
    return(power)