"""
Implements a DQN learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.utils import ReplayBuffer, Logger, TestMetric, set_global_seed
from src.envs.utils import ExtraAction

class SAC:
    """
    # Required parameters.
    envs : List of environments to use.
    network : Choice of neural network.
    # Initial network parameters.
    init_network_params : Pre-trained network to load upon initialisation.
    init_weight_std : Standard deviation of initial network weights.
    # DQN parameters
    double_dqn : Whether to use double DQN (DDQN).
    update_target_frequency : How often to update the DDQN target network.
    gamma : Discount factor.
    clip_Q_targets : Whether negative Q targets are clipped (generally True/False for irreversible/reversible agents).
    # Replay buffer.
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.
    # Learning rate
    update_learning_rate : Whether to dynamically update the learning rate (if False, initial_learning_rate is always used).
    initial_learning_rate : Initial learning rate.
    peak_learning_rate : The maximum learning rate.
    peak_learning_rate_step : The timestep (from the start, not from when training starts) at which the peak_learning_rate is found.
    final_learning_rate : The final learning rate.
    final_learning_rate_step : The timestep of the final learning rate.
    # Optional regularization.
    max_grad_norm : The norm grad to clip gradients to (None means no clipping).
    weight_decay : The weight decay term for regularisation.
    # Exploration
    update_exploration : Whether to update the exploration rate (False would tend to be used with NoisyNet layers).
    initial_exploration_rate : Inital exploration rate.
    final_exploration_rate : Final exploration rate.
    final_exploration_step : Timestep at which the final exploration rate is reached.
    # Loss function
    adam_epsilon : epsilon for ADAM optimisation.
    loss="mse" : Loss function to use.
    # Saving the agent
    save_network_frequency : Frequency with which the network parameters are saved.
    network_save_path : Folder into which the network parameters are saved.
    # Testing the agent
    evaluate : Whether to test the agent during training.
    test_envs : List of test environments.  None means the training environments (envs) are used.
    test_episodes : Number of episodes at each test point.
    test_frequency : Frequency of tests.
    test_save_path : Folder into which the test scores are saved.
    test_metric : The metric used to quantify performance.
    # Other
    logging : Whether to log.
    seed : The global seed to set.  None means randomly selected.
    """
    def __init__(
        self,
        envs,
        network,

        # Initial network parameters.
        init_network_params = None,
        init_weight_std = None,

        # DQN parameters
        double_dqn = True,
        update_target_frequency=10000,
        gamma=0.99,
        clip_Q_targets=False,

        # Replay buffer.
        replay_start_size=50000,
        replay_buffer_size=1000000,
        minibatch_size=32,
        update_frequency=1,

        # Learning rate
        update_learning_rate = True,
        initial_learning_rate = 0,
        peak_learning_rate = 1e-3,
        peak_learning_rate_step = 10000,
        final_learning_rate = 5e-5,
        final_learning_rate_step = 200000,

        # Optional regularization.
        max_grad_norm=None,
        weight_decay=0,

        # Exploration
        update_exploration=True,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,

        # Loss function
        adam_epsilon=1e-8,
        loss="mse",

        # Saving the agent
        save_network_frequency=10000,
        network_save_path='network',

        # Testing the agent
        evaluate=True,
        test_envs=None,
        test_episodes=20,
        test_frequency=10000,
        test_save_path='test_scores',
        test_metric=TestMetric.ENERGY_ERROR,

        # Other
        logging=True,
        seed=None
    ):

        self.device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.double_dqn = double_dqn

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay=weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration,
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        if type(envs)!=list:
            envs = [envs]
        self.envs = envs
        self.env, self.acting_in_reversible_spin_env  = self.get_random_env()

        self.replay_buffers = {}
        for n_spins in set([env.action_space.n for env in self.envs]):
            self.replay_buffers[n_spins] = ReplayBuffer(self.replay_buffer_size)

        self.replay_buffer = self.get_replay_buffer_for_env(self.env)

        self.seed = random.randint(0, 1e6) if seed is None else seed

        for env in self.envs:
            set_global_seed(self.seed, env)
        self.network_policy = network().to(self.device)
        self.networkQ1 = network().to(self.device)
        self.networkQ2 = network().to(self.device)
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params != None:
            print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std != None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)
                with torch.no_grad():
                    self.networkQ1.apply(init_weights)
                    self.networkQ2.apply(init_weights)
                    self.network_policy.apply(init_weights)

        self.target_networkQ1 = network().to(self.device)
        self.target_networkQ2 = network().to(self.device)
        self.target_networkQ1.load_state_dict(self.networkQ1.state_dict())
        self.target_networkQ2.load_state_dict(self.networkQ2.state_dict())
        for param in self.target_networkQ1.parameters():
            param.requires_grad = False
        for param in self.target_networkQ2.parameters():
            param.requires_grad = False

        self.optimizerQ1 = optim.Adam(self.networkQ1.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay, amsgrad = True)
        self.optimizerQ2 = optim.Adam(self.networkQ2.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay, amsgrad = True)
        self.optimizer_policy = optim.Adam(self.network_policy.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay, amsgrad = True)

        self.evaluate = evaluate
        if test_envs in [None,[None]]:
            # By default, test on the same environment(s) as are trained on.
            self.test_envs = self.envs
        else:
            if type(test_envs) != list:
                test_envs = [test_envs]
            self.test_envs = test_envs
        self.test_episodes = int(test_episodes)
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric
        
        self.q1losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "q1losses.pkl")
        self.q2losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "q2losses.pkl")
        self.policy_losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "policy_losses.pkl")
        self.entropy_losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "entropy_losses.pkl")
        if not self.acting_in_reversible_spin_env:
            for env in self.envs:
                assert env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."
            for env in self.test_envs:
                assert env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."

        self.allowed_action_state = self.env.get_allowed_action_states()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path
        
        
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        target_entropy_ratio = 0.98
        self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.initial_learning_rate, amsgrad = True)
        


    def get_random_env(self, envs=None):
        if envs is None:
            env = random.sample(self.envs, k=1)[0]
        else:
            env = random.sample(envs, k=1)[0]

        return env, env.reversible_spins

    def get_replay_buffer_for_env(self, env):
        return self.replay_buffers[env.action_space.n]

    def get_random_replay_buffer(self):
        return random.sample(self.replay_buffers.items(), k=1)[0][1]

    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger()

        # Initialise the state
        state = torch.as_tensor(self.env.reset())
        score = 0.
        losses_eps = []
        t1 = time.time()

        test_scores = []
        q1losses = []
        q2losses = []
        policy_losses = []
        entropy_losses = []
        

        is_training_ready = False

        for timestep in range(timesteps):

            if not is_training_ready:
                if all([len(rb)>=self.replay_start_size for rb in self.replay_buffers.values()]):
                    print('\nAll buffers have {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    is_training_ready=True

            # Choose action
            with torch.no_grad():
                action,_,_ = self.sample(state.to(self.device).float(), self.acting_in_reversible_spin_env)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate:
                self.update_lr(timestep)

            # Perform action in environment
            state_next, reward, done, _ = self.env.step(action.item())

            score += reward

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next)

            done = torch.as_tensor([done], dtype=torch.float)

            self.replay_buffer.add(state, action, reward, state_next, done)

            if done:
                # Reinitialise the state
                if verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep+1),
                         self.env.current_step,
                         np.round(score,3),
                         loss_str,
                         round(time.time() - t1, 3)))
                    
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                self.env, self.acting_in_reversible_spin_env = self.get_random_env()
                self.replay_buffer = self.get_replay_buffer_for_env(self.env)
                state = torch.as_tensor(self.env.reset())
                score = 0
                losses_eps = []
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:

                # Update the main network
                if timestep % self.update_frequency == 0:
                    
                    # Sample a batch of transitions
                    transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.device)
                    # Train on selected batch
                    q1loss, q2loss, policy_loss, entropy_loss = self.train_step(transitions)
                    q1losses.append([timestep, q1loss])
                    q2losses.append([timestep, q2loss])
                    policy_losses.append([timestep, policy_loss])
                    entropy_losses.append([timestep, entropy_loss])
                    losses_eps.append(q1loss)

                    if self.logging:
                        logger.add_scalar('Loss', loss, timestep)

                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_networkQ1.load_state_dict(self.networkQ1.state_dict())
                    self.target_networkQ2.load_state_dict(self.networkQ2.state_dict())

            if (timestep+1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                
                test_score = self.evaluate_agent()
                print('\nTest score: {}\n'.format(np.round(test_score,3)))

                if self.test_metric in [TestMetric.FINAL_CUT,TestMetric.MAX_CUT,TestMetric.CUMULATIVE_REWARD]:
                    best_network = all([test_score > score for t,score in test_scores])
                elif self.test_metric in [TestMetric.ENERGY_ERROR, TestMetric.BEST_ENERGY]:
                    best_network = all([test_score < score for t, score in test_scores])
                else:
                    raise NotImplementedError("{} is not a recognised TestMetric".format(self.test_metric))

                if best_network:
                    path = self.network_save_path
                    path_main, path_ext = os.path.splitext(path)
                    path_main += "_best"
                    if path_ext == '':
                        path_ext += '.pth'
                    self.save(path_main + path_ext)

                test_scores.append([timestep+1,test_score])

            if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
                path = self.network_save_path
                path_main, path_ext = os.path.splitext(path)
                path_main += str(timestep+1)
                if path_ext == '':
                    path_ext += '.pth'
                self.save(path_main+path_ext)

        if self.logging:
            logger.save()

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'

        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('test_scores saved to {}'.format(path))
                
        with open(self.q1losses_save_path, 'wb+') as output:
            pickle.dump(np.array(q1losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('q1losses saved to {}'.format(path))
                
        with open(self.q2losses_save_path, 'wb+') as output:
            pickle.dump(np.array(q2losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('q2losses saved to {}'.format(path))
        
        with open(self.policy_losses_save_path, 'wb+') as output:
            pickle.dump(np.array(policy_losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('policy_losses saved to {}'.format(path))
                
        with open(self.entropy_losses_save_path, 'wb+') as output:
            pickle.dump(np.array(entropy_losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('entropy_losses saved to {}'.format(path))

    @torch.no_grad()
    def __only_bad_actions_allowed(self, state, network):
        x = (state[0, :] == self.allowed_action_state).nonzero()
        q_next = network(state.to(self.device).float())[0][x].max()
        return True if q_next < 0 else False

    def reward_discount(self, rewards, dones, gamma):
        ##NOT THE BEST IMPLEMENTATION
        new_rewards = []
        r_rewards = torch.flip(rewards, (0,1)).squeeze(-1).cpu().tolist()
        r_dones = torch.flip(dones, (0,1)).squeeze(-1).cpu().tolist()
        temp_reward = 0
        for r,d in zip(r_rewards, r_dones):
            
            if d==1.0:
                temp_reward = 0
            r1 = r
            if isinstance(r,list):
                r1 = r[0]
            temp_reward = r1 + (gamma * temp_reward)
            new_rewards.insert(0, temp_reward)
        return new_rewards
    

    def reward_norm(self, rewards):
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        return rewards
    
    def train_step(self, transitions):

        states, actions, rewards, states_next, dones = transitions
        ##networks: self.network_policy; self.networkQ1 - online; self.networkQ2 - online ;self.target_networkQ1 -target ;self.target_networkQ2 - target
        
        #calc critic loss- input:batch; output:q1_loss, q2_loss, errors, mean_q1, mean_q2
        ##current q 
        curr_q1,_ = self.networkQ1(states.float())
        curr_q2,_ = self.networkQ2(states.float())

        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        ##target q
        with torch.no_grad():
            _,action_probs, log_action_probs = self.sample(states_next.float(), self.acting_in_reversible_spin_env, target_q=False)
            next_q1,_ = self.target_networkQ1(states_next.float())
            next_q2,_ = self.target_networkQ2(states_next.float())
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=-1, keepdim=True)#sum?
        assert rewards.shape == next_q.shape
        target_q = rewards + (1.0 - dones) * self.gamma * next_q
        ##td errors
        errors = torch.abs(curr_q1.detach() - target_q)
        ##log means for tracking
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
        ##actial critic loss
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))
        
        #calc policy loss- input:batch; output: policy_loss, entropies
        ##sampling from the policy: log_probs, probs of actions

        _, action_probs, log_action_probs = self.sample(states, self.acting_in_reversible_spin_env, target_q = True)
        ##torch no grad with online critic: input:samples; output:q
        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            _,q1 = self.networkQ1(states.float())
            _,q2 = self.networkQ2(states.float())
            q = torch.min(q1, q2)

        ##expectation of entropies
        entropies = -torch.sum(action_probs * log_action_probs, dim=-1, keepdim=True)

        ##expectation of Q
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=-1, keepdim=True)
        ##actual policy loss
        policy_loss = (- q - self.alpha * entropies).mean()
        entropies = entropies.detach()
        
        #cals entropy - input:entropies; output:entropy_loss
        #entropy_loss
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))
        
        # Update weights
        #q1
        self.optimizerQ1.zero_grad()
        q1_loss.backward()
        self.optimizerQ1.step()
        #q2
        self.optimizerQ2.zero_grad()
        q2_loss.backward()
        self.optimizerQ2.step()
        #policy
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        #alpha
        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step() 
        #update alpha
        self.alpha = self.log_alpha.exp()
        
        #if self.max_grad_norm is not None: #Optional gradient clipping
        #    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        
        
#         if self.acting_in_reversible_spin_env:
#             # Calculate target Q
#             with torch.no_grad():
#                 if self.double_dqn:
#                     greedy_actions = self.network(states_next.float())[0].argmax(1, True)
#                     q_value_target = self.target_network(states_next.float())[0].gather(1, greedy_actions)
#                 else:
#                     q_value_target = self.target_network(states_next.float())[0].max(1, True)[0]

#         else:
#             target_preds = self.target_network(states_next.float())[0]
#             disallowed_actions_mask = (states_next[:, 0, :] != self.allowed_action_state)
#             # Calculate target Q, selecting ONLY ALLOWED ACTIONS greedily.
#             with torch.no_grad():
#                 if self.double_dqn:
#                     network_preds = self.network(states_next.float())[0]
#                     # Set the Q-value of disallowed actions to a large negative number (-10000) so they are not selected.
#                     network_preds_allowed = network_preds.masked_fill(disallowed_actions_mask,-10000)
#                     greedy_actions = network_preds_allowed.argmax(1, True)
#                     q_value_target = target_preds.gather(1, greedy_actions)
#                 else:
#                     q_value_target = target_preds.masked_fill(disallowed_actions_mask,-10000).max(1, True)[0]

#         if self.clip_Q_targets:
#             q_value_target[q_value_target < 0] = 0

#         # Calculate TD target
#         td_target = rewards + (1 - dones) * self.gamma * q_value_target

#         # Calculate Q value
#         q_value = self.network(states.float())[0].gather(1, actions)

#         # Calculate loss
#         loss = self.loss(q_value, td_target, reduction='mean')

#         # Update weights
#         self.optimizer.zero_grad()
#         loss.backward()

        if self.max_grad_norm is not None: #Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.networkQ1.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.networkQ2.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.target_networkQ1.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.target_networkQ2.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.network_policy.parameters(), self.max_grad_norm)


#         self.optimizer.step()

        return mean_q1, mean_q2, policy_loss.item(), entropy_loss.item()
    
    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximises Q function
            action = self.predict(state)
        else:
            if self.acting_in_reversible_spin_env:
                # Random random spin.
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Flip random spin from that hasn't yet been flipped.
                x = (state[0, :] == self.allowed_action_state).nonzero()
                action = x[np.random.randint(0, len(x))].item()
        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
                )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (self.final_learning_rate_step - self.peak_learning_rate_step)
                )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
    def sample(self, states, acting_in_reversible_spin_env=None, target_q = False):
        
        logits,_ = self.network_policy(states.to(self.device).float())
        action_probs = torch.nn.functional.softmax(logits, dim = -1)
        if acting_in_reversible_spin_env:
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()#.view(-1, 1)
        else:
            if action_probs.dim() == 1:
                x = (states[0, :] == self.allowed_action_state)#.nonzero()
                actions_probs = action_probs.masked_fill(x, 0.)#.item()
            else:
                if target_q:
                    disallowed_actions_mask = (states[:, 0, :] != self.allowed_action_state)
                else:
                    disallowed_actions_mask = (states[:, :, 0] != self.allowed_action_state)
                action_probs = action_probs.masked_fill(disallowed_actions_mask, 0.)
            # Avoid numerical instability.

            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        
        return actions, action_probs, log_action_probs
    @torch.no_grad()
    def predict(self, states, acting_in_reversible_spin_env=None):
        
        if acting_in_reversible_spin_env is None:
            acting_in_reversible_spin_env = self.acting_in_reversible_spin_env

        d_inputs, _ = self.network_policy(states)
        if acting_in_reversible_spin_env:
            #actions = distribution.mean
            if d_inputs.dim() == 1:
                actions = d_inputs.argmax()#.item()
            else:
                actions = d_inputs.argmax(1, True).squeeze(1)#.cpu().numpy()
            return actions
        else:
            #TO CHANGE LATER
            if d_inputs.dim() == 1:
                x = (states[0, :] == self.allowed_action_state).nonzero()
                actions = x[d_inputs[x].argmax().item()].item()
            else:
                disallowed_actions_mask = (states[:, :, 0] != self.allowed_action_state)
                d_inputs_allowed = d_inputs.masked_fill(disallowed_actions_mask, 0)
                actions = d_inputs_allowed.argmax(1, True).squeeze(1)#.cpu().numpy()
            return actions

    @torch.no_grad()
    def evaluate_agent(self, batch_size=None):
        """
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        """
        if batch_size is None:
            batch_size = self.minibatch_size

        i_test = 0
        i_comp = 0
        test_scores = []
        batch_scores = [0]*batch_size

        test_envs = np.array([None]*batch_size)
        obs_batch = []

        while i_comp < self.test_episodes:

            for i, env in enumerate(test_envs):
                if env is None and i_test < self.test_episodes:
                    test_env, testing_in_reversible_spin_env = self.get_random_env(self.test_envs)
                    obs = test_env.reset()
                    test_env = deepcopy(test_env)

                    test_envs[i] = test_env
                    obs_batch.append(obs)

                    i_test += 1

            actions = self.predict(torch.FloatTensor(np.array(obs_batch)).to(self.device),
                                   testing_in_reversible_spin_env)
            actions = actions.cpu().numpy()
            obs_batch = []

            i = 0
            for env, action in zip(test_envs, actions):

                if env is not None:
                    obs, rew, done, info = env.step(action)

                    if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                        batch_scores[i] += rew

                    if done:
                        if self.test_metric == TestMetric.BEST_ENERGY:
                            batch_scores[i] = env.best_energy
                        elif self.test_metric == TestMetric.ENERGY_ERROR:
                            batch_scores[i] = abs(env.best_energy - env.calculate_best()[0])
                        elif self.test_metric == TestMetric.MAX_CUT:
                            batch_scores[i] = env.get_best_cut()
                        elif self.test_metric == TestMetric.FINAL_CUT:
                            batch_scores[i] = env.calculate_cut()

                        test_scores.append(batch_scores[i])

                        if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                            batch_scores[i] = 0

                        i_comp += 1
                        test_envs[i] = None
                    else:
                        obs_batch.append(obs)

                i += 1

        if self.test_metric == TestMetric.ENERGY_ERROR:
            print("\n{}/{} graphs solved optimally".format(np.count_nonzero(np.array(test_scores)==0),self.test_episodes), end="")

        return np.mean(test_scores)

    def save(self, path='network.pth'):
        if os.path.splitext(path)[-1]=='':
            path + '.pth'
        torch.save(self.network_policy.state_dict(), path)

    def load(self,path):
        self.network_policy.load_state_dict(torch.load(path,map_location=self.device))