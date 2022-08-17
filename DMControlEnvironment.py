import numpy as np
import copy
from dm_control import suite


DATA_ATTR = ['qacc_smooth', 'qacc_warmstart', 'qacc_warmstart', 'qfrc_actuator', 'qfrc_applied', 'qfrc_bias', 'qfrc_constraint', 'qfrc_inverse', 'qfrc_passive', 'qfrc_smooth']
             

class np_with_data(list):
    'class of float number with inner attribute i'
    def __init__(self, v):
        super().__init__(v)
        self.data = None
        self.shape = v.shape
        return None


class DMControlEnvironment():
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
        #state = np_with_data(self.env.physics.state())
        #state.data = self.get_data()
        self.state = self.env.physics.state()
        return self.state
    
    def step(self, action):
        step = self.env.step(action)
        #state = np_with_data(self.env.physics.state())
        self.state = self.env.physics.state()
        #state.data = self.get_data()
        return self.state, step.reward, False, {}
    
    def virtual_step(self, state, action):
        #self.env.reset()
        #self.env.physics.reset()
        #print(state.data)
        self.env.physics.set_state(state)
        
        #for attr_name in ['qacc_smooth']:
        #    print(attr_name)
        #    print(state.data[attr_name])
        #print(state.data)
        #for attr_name in DATA_ATTR:
        #    setattr(self.env.physics.data, attr_name, state.data[attr_name])
                
        
        step = self.env.step(action)
        state = self.env.physics.state()
        #state = np_with_data(self.env.physics.state())
        #state.data = self.get_data()
        return state, step.reward, False, {}
    
    
    def get_data(self):
        data = {}
        for attr_name in DATA_ATTR:
            data[attr_name] = getattr(self.env.physics.data, attr_name)
        return data

        
      