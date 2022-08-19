import gym, os
import numpy as np
import pathlib


class NPArrayWithPhysics(np.ndarray):
    def __new__(cls, input_array, physics=None):
        obj = np.asarray(input_array).view(cls)
        obj.physics = physics
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.physics = getattr(obj, 'physics', None)



class GymEnv:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        
        if domain_name in ['Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 
                           'Swimmer-v3', 'Walker2d-v3']:
            self.env = gym.make(domain_name, reset_noise_scale=0)
        else:
            self.env = gym.make(domain_name)
        self.env.seed(21)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_min = self.env.action_space.low
        self.action_max = self.env.action_space.high
        self.pos_dim = {'InvertedDoublePendulum-v2':3, 'InvertedPendulum-v2':2}
        return None
        
    def reset(self):
        if self.domain_name in ['InvertedDoublePendulum-v2', 'InvertedPendulum-v2']:
            self.env.reset()
            state = np.zeros(self.state_dim)
            sim_state = self.env.sim.get_state()
            for i in range(self.pos_dim[self.domain_name]):
                sim_state.qpos[i] = 0
                sim_state.qvel[i] = 0
            state = NPArrayWithPhysics(state, physics=sim_state)
            self.env.sim.set_state(state.physics)
            return state
            
        else:
            state = self.env.reset()
            state = NPArrayWithPhysics(state, self.env.sim.get_state())
            return state
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = NPArrayWithPhysics(state, self.env.sim.get_state())
        return state, reward, done, info
    
    def virtual_step(self, state, action):
        self.env.sim.set_state(state.physics)
        state, reward, done, info = self.env.step(action)
        state = NPArrayWithPhysics(state, self.env.sim.get_state())
        return state, reward, done, info 
    
    
        