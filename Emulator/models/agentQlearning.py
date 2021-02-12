import numpy as np
import random

class AgentSarsa(object):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.8):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.actions = env.action_space
    
    def get_q_table(self):
        return self.q_table
    
    def set_q_table(self, q_table):
        self.q_table=q_table
        
    def set_epsilon(self, epsilon):
        self.epsilon=epsilon
    
    def check_if_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state]=np.zeros(self.actions.n)
            
    def get_action(self, state):
        self.check_if_state_exist(state)
        if np.random.rand() > self.epsilon:
            target_actions = self.q_table[state]
            idx_list=list(range(len(target_actions)))
            random.shuffle(idx_list)
            reordered=target_actions[idx_list]
            target_action = idx_list[np.argmax(reordered)]
        else:
            target_action = self.actions.sample()
        return target_action

    def update_q_table(self, state, action, reward, state_next, terminal):
        self.check_if_state_exist(state_next)
        q_value_predict = self.q_table[state][action]
        if terminal == False:
            q_value_real = reward + self.gamma * np.amax(self.q_table[state_next])
        else:
            q_value_real = reward
        self.q_table[state][action] += self.alpha * (q_value_real - q_value_predict)

    def train(self, state):
        # Get first action.
        action = self.get_action(state)
        # Get next state.
        state_next, reward, terminal, info = self.env.step(action)
        # Get next action.
        action_next = self.get_action(state_next)
        # Update Q table.
        self.update_q_table(state, action, reward, state_next, action_next, terminal)
        return state_next, reward, terminal 