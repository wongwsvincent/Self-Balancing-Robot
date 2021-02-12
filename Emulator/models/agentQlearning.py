import numpy as np
import random
import math

class AgentQlearning(object):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_lifetime=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_lifetime = epsilon_lifetime
        self.q_table = {}
        self.actions = env.action_space
        self.visits = {}
        self.training = True
    
    def get_q_table(self):
        return self.q_table
    
    def set_q_table(self, q_table):
        self.q_table=q_table
        
    def set_epsilon(self, epsilon):
        self.epsilon=epsilon

    def set_epsilon_lifetime(self, epsilon_lifetime):
        self.epsilon_lifetime=epsilon_lifetime
    
    def train_mode(self, flag):
        self.training=flag

    def check_if_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state]=np.zeros(self.actions.n)
            self.visits[state]=0
            
    def get_action(self, state):
        self.check_if_state_exist(state)
        if self.training==True and np.random.rand() < 0.5*(math.exp(-self.visits[state]/self.epsilon_lifetime)+1)*self.epsilon:
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
        # Get next action.
        action_next = self.get_action(state_next)
        # Update Q table.
        self.update_q_table(state, action, reward, state_next, terminal)
        return state_next, reward, terminal 