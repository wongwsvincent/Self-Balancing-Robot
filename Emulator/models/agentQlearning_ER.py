import numpy as np
import random
import math
from collections import deque

class AgentQlearning_ER(object):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_max=1.0, epsilon_min=0.2, epsilon_halflife=10, memory_size=100000, memory_sampling=10, update_freq=100):
        self.env = env
        self.actions = env.action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_halflife = epsilon_halflife
        self.q_table = {}
        self.visits = {}
        self.training = True
        self.memory_size = memory_size
        self.memory_sampling = memory_sampling
        self.memory_table = deque(maxlen=self.memory_size)
        self.update_freq = 100
        self.counter = 0
    
    def get_q_table(self):
        return self.q_table
    
    def set_q_table(self, q_table):
        self.q_table=q_table
        
    def set_epsilon_max(self, epsilon_max):
        self.epsilon_max=epsilon_max

    def set_epsilon_min(self, epsilon_min):
        self.epsilon_min=epsilon_min
    
    def set_epsilon_halflife(self, epsilon_halflife):
        self.epsilon_halflife=epsilon_halflife
    
    def train_mode(self, flag):
        self.training=flag

    def check_if_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state]=np.zeros(self.actions.n)
            self.visits[state]=0.
            
    def get_action(self, state):
        self.check_if_state_exist(state)
        if self.training==True and np.random.rand() < self.epsilon_min + (self.epsilon_max-self.epsilon_min)*pow(0.5, self.visits[state]/self.epsilon_halflife):
            target_action = self.actions.sample()
        else:
            target_actions = self.q_table[state]
            idx_list=list(range(len(target_actions)))
            random.shuffle(idx_list)
            reordered=target_actions[idx_list]
            target_action = idx_list[np.argmax(reordered)]
        return target_action

    def update_q_table(self, state, action, reward, state_next, terminal):
        self.check_if_state_exist(state_next)
        q_value_predict = self.q_table[state][action]
        if terminal == False:
            q_value_real = reward + self.gamma * np.amax(self.q_table[state_next])
        else:
            q_value_real = reward
        self.q_table[state][action] += self.alpha * (q_value_real - q_value_predict)
        self.visits[state] += 1

    def train(self, state):
        # Get first action.
        action = self.get_action(state)
        # Get next state.
        state_next, reward, terminal, info = self.env.step(action)
        # Store transition experience in memory
        self.memory_table.append((state, action, reward, state_next, terminal))
        self.counter+=1
        if (self.counter%self.update_freq==0):
            # Sample random minibatch of transition experiences from memory
            experiences = np.random.choice(len(self.memory_table), size=min(len(self.memory_table),self.memory_sampling), replace=False)
            # Update Q table with past experiences
            for exp_idx in experiences:
                self.update_q_table(self.memory_table[exp_idx][0], self.memory_table[exp_idx][1], self.memory_table[exp_idx][2], self.memory_table[exp_idx][3], self.memory_table[exp_idx][4])
            # Update Q table with the latest experience
            self.update_q_table(state, action, reward, state_next, terminal)
            self.counter=0
        return state_next, reward, terminal 
