import numpy as np
import copy
from dm_control import suite


class NPArrayWithPhysics(np.ndarray):
    def __new__(cls, input_array, physics=None):
        obj = np.asarray(input_array).view(cls)
        obj.physics = physics
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.physics = getattr(obj, 'physics', None)


class DMControlEnvWithPhysics():
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': np.random.RandomState(21)})
        self.state_dim = self.env.physics.state().shape[0]
        self.action_dim = self.env.action_spec().shape[0]
        self.action_min = self.env.action_spec().minimum
        self.action_max = self.env.action_spec().maximum
        self.attrs = []
        for attr_name in dir(self.env.physics.data):
            attr = getattr(self.env.physics.data, attr_name)
            if type(attr) is np.ndarray or type(attr) is int or type(attr) is float:
                self.attrs.append(attr_name)
        
        return None
        
    def reset(self):
        self.env.reset()
        self.env.physics.reset()
        state = NPArrayWithPhysics(self.env.physics.state(), physics=self.get_physics())
        self.state = state
        return self.state
    
    def step(self, action):
        step = self.env.step(action)
        state = NPArrayWithPhysics(self.env.physics.state(), physics=self.get_physics())
        self.state = state
        return state, step.reward, step.last(), {}
    
    def virtual_step(self, state, action):
        self.env.physics.set_state(state)
        for attr_name in self.attrs:
            setattr(self.env.physics.data, attr_name, state.physics[attr_name])
                
        step = self.env.step(action)
        state = NPArrayWithPhysics(self.env.physics.state(), physics=self.get_physics())
        return state, step.reward, step.last(), {}
    
    def get_physics(self):
        physics = {}
        for attr_name in self.attrs:
            attr = copy.copy(getattr(self.env.physics.data, attr_name))
            physics[attr_name] = attr
        return physics
