"""
Trains and tests S2V-DQN on 20 spin ER graphs.
"""
import experiments.ER_20spin.test.test_s2v_ppo as test
import experiments.ER_20spin.train.train_s2v_ppo as train

save_loc="ER_20spin/s2v_ppo32"

train.run(save_loc, minibatch_size = 32)

test.run(save_loc, graph_save_loc="_graphs/validation/ER_20spin_p15_100graphs.pkl", batched=True, max_batch_size=None)
test.run(save_loc, graph_save_loc="_graphs/validation/ER_40spin_p15_100graphs.pkl", batched=True, max_batch_size=None)
test.run(save_loc, graph_save_loc="_graphs/validation/ER_60spin_p15_100graphs.pkl", batched=True, max_batch_size=None)
test.run(save_loc, graph_save_loc="_graphs/validation/ER_100spin_p15_100graphs.pkl", batched=True, max_batch_size=None)
test.run(save_loc, graph_save_loc="_graphs/validation/ER_200spin_p15_100graphs.pkl", batched=True, max_batch_size=25)
test.run(save_loc, graph_save_loc="_graphs/validation/ER_500spin_p15_100graphs.pkl", batched=True, max_batch_size=5)