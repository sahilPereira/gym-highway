# computation config
num_workers = 6
num_envs_per_worker = 1
num_gpus = 1

# training config
sample_batch_size = 64
train_batch_size = 1280
entropy_coeff = 0.1
use_lstm = False
episodes = 2000
initial_reward = -140.0

# policy network
fcnet_hiddens = [256, 256, 256]
fcnet_activation = "relu"
