import numpy as np
from dm_control import suite


class DMControlEnv():
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': np.random.RandomState(21)})
        self.state_dim = self.env.physics.state().shape[0]
        self.action_dim = self.env.action_spec().shape[0]
        self.action_min = self.env.action_spec().minimum
        self.action_max = self.env.action_spec().maximum
        return None
        
    def reset(self):
        self.env.reset()
        self.env.physics.reset()
        self.state = self.env.physics.state()
        return self.state
    
    def step(self, action):
        step = self.env.step(action)
        self.state = self.env.physics.state()
        return self.state, step.reward, False, {}
    
    def virtual_step(self, state, action):
        self.env.physics.set_state(state)
        step = self.env.step(action)
        state = self.env.physics.state()
        return state, step.reward, False, {}
      