import os, sys
import numpy as np
import copy
from dm_control import suite
sys.path.insert(0, os.path.abspath('..'))
from DMControl.DMControlEnv import DMControlEnv

PREFIXES = ['efc_AR','D_','efc_J_', 'efc_JT', 'efc_vel', 'efc_state', 'efc_margin', 
            'efc_frictionloss', 'efc_b','efc_force', 'cfrc_ext', 'cfrc_int', 'qDeriv', 
            'qLU', 'xaxis', 'xfrc_applied', 'actuator_moment', 'cacc', 'cam_xmat']


class NPArrayWithPhysics(np.ndarray):
    def __new__(cls, input_array, physics=None):
        obj = np.asarray(input_array).view(cls)
        obj.physics = physics
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.physics = getattr(obj, 'physics', None)


class DMControlEnvWithPhysics(DMControlEnv):
    def __init__(self, domain_name, task_name, dt):
        super().__init__(domain_name, task_name, dt)
        self.domain_name = domain_name
        self.attrs = self.get_attributes()
        return None
    
    def reset(self):
        state = self._reset()
        state = NPArrayWithPhysics(state, physics=self.get_physics())
        self.state = state
        return self.state
    
    def step(self, action):
        state, reward, done, info = self._step(action)
        state = NPArrayWithPhysics(state, physics=self.get_physics())
        self.state = state
        return self.state, reward, done, info
    
    def virtual_step(self, state, action):
        self.reset()
        self.env.physics.set_state(state)
        self.set_physics(state)
        state, reward, done, info = self._step(action)
        state = NPArrayWithPhysics(state, physics=self.get_physics())
        return state, reward, done, info
    
    def get_partial_attributes(self):
        #attrs = []
        #for attr_name in dir(self.env.physics.data):
        #   attr = getattr(self.env.physics.data, attr_name)
        #    if type(attr) is np.ndarray or type(attr) is int or type(attr) is float:
        #        attrs.append(attr_name)
        attrs = []
        for attr_name in dir(self.env.physics.data):
            ok = False
            for prefix in ['actuator_']:
                if prefix in attr_name:
                    ok = True
            if ok:
                attrs.append(attr_name)
            
        return attrs
    
    def get_attributes(self):
        attrs = []
        for attr_name in dir(self.env.physics.data):
            attr = getattr(self.env.physics.data, attr_name)
            if type(attr) is np.ndarray or type(attr) is int or type(attr) is float:
                if self.domain_name in ['finger', 'manipulator']:
                    attrs.append(attr_name)
                else:
                    if not any(prefix in attr_name for prefix in PREFIXES):
                        attrs.append(attr_name)
        return attrs
    
    def get_physics(self):
        physics = {}
        for attr_name in self.attrs:
            attr = copy.copy(getattr(self.env.physics.data, attr_name))
            physics[attr_name] = attr
        return physics
    
    def set_physics(self, state):
        for attr_name in self.attrs:
            setattr(self.env.physics.data, attr_name, state.physics[attr_name])
        return None
