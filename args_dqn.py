from src.agents.dqn.utils import TestMetric
args = {
"init_network_params":None,
"init_weight_std":0.01,

"double_dqn":True,
"clip_Q_targets":False,

"replay_start_size":500,
"replay_buffer_size":5000,  # 20000
"update_target_frequency":1000,  # 500

"update_learning_rate":False,
"initial_learning_rate":1e-5,
"peak_learning_rate":1e-5,
"peak_learning_rate_step":20000,
"final_learning_rate":1e-5,
"final_learning_rate_step":200000,

"update_frequency":32,  # 1
"minibatch_size":32,  # 128
"max_grad_norm":0.01,
"weight_decay":0,

"update_exploration":True,
"initial_exploration_rate":1,
"final_exploration_rate":0.05,  # 0.05
"final_exploration_step":150000,  # 40000

"adam_epsilon":1e-8,
"logging":False,
"loss":"mse",

"save_network_frequency":100000,

"evaluate":True,
    
"test_frequency":10000,  # 10000
"test_metric":TestMetric.MAX_CUT,

"seed":None}