import numpy as np

# An agent that uses random search
class AgentRandomSearch(object):
    def __init__(self, env):
        self.env = env
        self.parameters=None
        
    def get_params(self):
        return self.parameters
    
    def set_params(self, parameters):
        self.parameters=parameters
        
    def get_action(self, states):
        return 0 if np.matmul(self.parameters, states) < 0 else 1
    
    def train(self, state):
        self.parameters=np.random.rand(4) * 2 - 1
        action = self.get_action(state)
        obs_next, reward, terminal, info= self.env.step(action)
        return obs_next, reward, terminal