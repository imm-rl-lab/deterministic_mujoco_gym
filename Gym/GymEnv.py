import gym, os
import numpy as np
import pathlib
import os
os.add_dll_directory(os.path.join(os.path.expanduser('~'), ".mujoco", "mjpro150", "bin"))
from mujoco_py import GlfwContext
GlfwContext(True)

class NPArrayWithPhysics(np.ndarray):
    def __new__(cls, input_array, physics=None):
        obj = np.asarray(input_array).view(cls)
        obj.physics = physics
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.physics = getattr(obj, 'physics', None)

class GymEnv:
    def __init__(self, domain_name, dt):
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
        self.max_episode_steps = self.env._max_episode_steps
        
        self.pos_dim = {'InvertedDoublePendulum-v2':3, 'InvertedPendulum-v2':2}
        
        self.inner_dt = self.env.dt / self.env.env.frame_skip
        self.env.env.frame_skip = max(1, int(dt / self.inner_dt + 1e-6))
        self.dt = self.env.env.frame_skip * self.inner_dt
        
        if self.dt != dt:
            print('Warning: the closest possible dt is ', self.dt)
        
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
        self.reset()
        self.env.sim.set_state(state.physics)
        state, reward, done, info = self.env.step(action)
        state = NPArrayWithPhysics(state, self.env.sim.get_state())
        return state, reward, done, info 
    
    
        