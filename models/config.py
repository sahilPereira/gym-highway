# computation config
num_workers = 6
num_envs_per_worker = 1
num_gpus = 1
num_timesteps = 2e6

# ray training config
sample_batch_size = 64 # Size of batches collected from each worker
train_batch_size = 1280 # Number of timesteps collected for each SGD round
sgd_minibatch_size = 128 # Total SGD batch size across all devices for SGD
entropy_coeff = 0.1
use_lstm = False
episodes = 2000
initial_reward = -140.0
lr = 1e-4

# baselines ppo2 training config
ppo2_train_args = {'nsteps':240, #240,  1440 num_batches / 6 num_envs
				   'ent_coef':0.01, # NOTE: entropy regurarlization constant, this should range between 0-0.01 
				   'lr':1e-4,
				   'vf_coef':0.5, # NOTE: this was set to 1 in ray, but since vf loss is high, keep it at 0.5
				   'max_grad_norm': 60.0, # 0.5, # NOTE: should really increase this, since the grads are being clipped at 0.5
				   'gamma':0.99, 
				   'lam':1.0, # 0.95, # NOTE: this was set to 1 in ray
				   'log_interval':1, # there are ~ 694 updates for 1M timesteps
				   'nminibatches':6, # nbatch_train = nbatch // nminibatches = 1440//4 = 360
				   'noptepochs':1, # reduce epochs so that we dont use same old data to train model
				   'cliprange':0.2, # NOTE: ray default 0.2. Ray: [vf_clip_param, clip_param], baselines: [clip_param]
				   'save_interval':100, 
				   'load_path':None, 
				   'model_fn':None,
				   # policy network_kwargs
				   'num_layers':3, 
				   'num_hidden':256, 
				   'activation':'relu', 
				   'layer_norm':False
				   }

# Whether to place workers on GPUs (only for A3C)
use_gpu_for_workers = False

# intrinsic curiosity module parameters
use_icm = False

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
tensorboard_rootdir = 'gym_highway/tb/'
tensorboard_save_graph = True