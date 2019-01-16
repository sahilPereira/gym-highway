# computation config
num_workers = 6
num_envs_per_worker = 1
num_gpus = 1

# training config
sample_batch_size = 64 # Size of batches collected from each worker
train_batch_size = 1280 # Number of timesteps collected for each SGD round
sgd_minibatch_size = 128 # Total SGD batch size across all devices for SGD
entropy_coeff = 0.1
use_lstm = False
episodes = 2000
initial_reward = -140.0
lr = 1e-4
ppo2_train_args = {'nsteps':214,  # 1284 num_batches / 6 num_envs
				   'ent_coef':0.1, 
				   'lr':1e-4,
				   'vf_coef':0.5, 
				   'max_grad_norm':0.5, 
				   'gamma':0.99, 
				   'lam':0.95,
				   'log_interval':1, # there are ~ 778 updates for 1M timesteps
				   'nminibatches':4, # nbatch_train = nbatch // nminibatches = 1284//4 ~= 321
				   'noptepochs':5, 
				   'cliprange':0.2,
				   'save_interval':10, 
				   'load_path':None, 
				   'model_fn':None
				   }

# Whether to place workers on GPUs (only for A3C)
use_gpu_for_workers = False

# policy network
fcnet_hiddens = [256, 256, 256]
fcnet_activation = "relu"

# gym environment register
env_id = 'Highway-train-v0'
env_entry_point = 'gym_highway.envs:HighwayEnv'
env_train_kwargs = {'manual': False, 'inf_obs': True, 'save': False, 'render': False}
env_play_kwargs = {'manual': False, 'inf_obs': True, 'save': False, 'render': True}
results_dir = '~/gym_highway_results'
save_path = '~/gym_highway_results/ppo2_test1'

# logging
baselines_log_format = ['stdout','tensorboard']