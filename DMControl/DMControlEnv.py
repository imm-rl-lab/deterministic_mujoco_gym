import numpy as np
from dm_control import suite


class DMControlEnv():
    def __init__(self, domain_name, task_name, dt):
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': np.random.RandomState(21)})
        self.state_dim = self.env.physics.state().shape[0]
        self.action_dim = self.env.action_spec().shape[0]
        self.action_min = self.env.action_spec().minimum
        self.action_max = self.env.action_spec().maximum
        self.inner_dt = self.env.physics.timestep()
        self.inner_step_limit =  self.env._step_limit
        
        self.inner_step_n = max(int(dt / self.inner_dt + 1e-6), 1)
        self.dt = self.inner_step_n * self.inner_dt
        
        if self.dt != dt:
            print('Warning: the closest possible dt is ', self.dt)
        
        return None
        
    def reset(self):
        self.state = self._reset()
        return self.state
    
    def _reset(self):
        self.env.reset()
        self.env.physics.reset()
        return self.env.physics.state()
    
    def step(self, action):
        state, reward, done, info = self._step(action)
        self.state = state
        return self.state, reward, done, info

    def _step(self, action):
        reward = 0
        done = False
        for _ in range(self.inner_step_n):
            step = self.env.step(action)
            reward += step.reward
            done = max(done, step.last())
            if done:
                break
        
        if done and self.env.physics.time() > self.inner_dt * (self.inner_step_limit - 1):
            print('Error: done=True because of env.inner_step_limit')
                
        state = self.env.physics.state()
        return state, reward, done, {}
    
    def virtual_step(self, state, action):
        self.reset()
        self.env.physics.set_state(state)
        state, reward, done, info = self._step(action)
        return state, reward, done, info
    
