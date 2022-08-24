# Mujoco Gym

In this repository, we are trying different ways to make reinforcement learning environments from [Mujoco Gym](https://www.gymlibrary.ml/environments/mujoco/) and [dm_control](https://github.com/deepmind/dm_control) deterministic. We strive to ensure that the environments have the following important properties:

- the function <code>reset()</code> gives always the same initial state;
- the function <code>step(action)</code> such that a sequence of <code>actions</code> uniquely determines <code>states</code> and <code>rewards</code>;
- the function <code>virtual_step(state, action)</code> uniquely determines <code>next_state</code> and <code>reward</code>.

We implemented a wrapper for the environments that aims to fulfill these points, but unfortunately, it works not for all environments yet. 
The main reason is that Mujoco has internal variables and structures to which there is no obvious access. For example, in many environments, <code>next_state</code> and <code>reward</code> depend not only on the current <code>states</code>, but also on "internal physics". Nonetheless, we strive to take such things into account. Our results are described in the table below.

Environments                                          |    reset()    |    step()    |   virtual_step()    	
----------------------------------------------------- | ------------- | ------------ | ----------------
DMControlEnv('acrobot', 'swingup')                    |       0       |       0      |        0
DMControlEnv('acrobot', 'swingup_sparse')             |       0       |       0      |        0
DMControlEnv('ball_in_cup', 'catch')                  |       0       |   $10^{-2}$  |       $10^{-2}$
DMControlEnvWithPhysics('ball_in_cup', 'catch')       |       0       |       0      |        0
DMControlEnv('cartpole', 'balance')                   |       0       |   $10^{-14}$ |       $10^{-14}$
DMControlEnvWithPhysics('cartpole', 'balance')        |       0       |       0      |        0
DMControlEnv('cartpole', 'balance_sparse')            |       0       |   $10^{-14}$ |       $10^{-14}$
DMControlEnvWithPhysics('cartpole', 'balance_sparse') |       0       |       0      |        0
DMControlEnv('cartpole', 'swingup')                   |       0       |   $10^{-14}$ |       $10^{-14}$
DMControlEnvWithPhysics('cartpole', 'swingup')        |       0       |       0      |        0
DMControlEnv('cartpole', 'swingup_sparse')            |       0       |   $10^{-14}$ |       $10^{-14}$
DMControlEnvWithPhysics('cartpole', 'swingup_sparse') |       0       |       0      |        0
DMControlEnv('cheetah', 'run')                        |       0       |   $10^{2}$   |       $10^{1}$
DMControlEnvWithPhysics('cheetah', 'run')             |       0       |       0      |        0
DMControlEnv('finger', 'spin')                        |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('finger', 'spin')             |       0       |       0      |        0
DMControlEnv('finger', 'turn_easy')                   |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('finger', 'turn_easy')        |       0       |       0      |        0
DMControlEnv('finger', 'turn_hard')                   |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('finger', 'turn_hard')        |       0       |       0      |        0
DMControlEnv('fish', 'upright')                       |       0       |   $10^{0}$   |       $10^{0}$
DMControlEnvWithPhysics('fish', 'upright')            |       0       |       0      |        0
DMControlEnv('fish', 'swim')                          |   $10^{-3}$   |   $10^{0}$   |       $10^{0}$
DDMControlEnvWithPhysics('fish', 'swim')              |   $10^{-4}$   |   $10^{0}$   |       $10^{-0}$
DMControlEnv('hopper', 'stand')                       |       0       |   $10^{0}$   |       $10^{-1}$
DMControlEnvWithPhysics('hopper', 'stand')            |       0       |       0      |        0
DMControlEnv('hopper', 'hop')                         |       0       |   $10^{0}$   |       $10^{-1}$
DMControlEnvWithPhysics('hopper', 'hop')              |       0       |       0      |        0
DMControlEnv('humanoid', 'stand')                     |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('humanoid', 'stand')          |       0       |       0      |        0
DMControlEnv('humanoid', 'walk')                      |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('humanoid', 'walk')           |       0       |       0      |        0
DMControlEnv('humanoid', 'run')                       |       0       |   $10^{1}$   |       $10^{0}$
DMControlEnvWithPhysics('humanoid', 'run')            |       0       |       0      |        0
DMControlEnv('manipulator', 'bring_ball')             |   $10^{-58}$  |   $10^{0}$   |       $10^{0}$
DMControlEnvWithPhysics('manipulator', 'bring_ball')  |   $10^{-59}$  |   $10^{-29}$ |       $10^{-29}$
DMControlEnv('pendulum', 'swingup')                   |       0       |   $10^{2}$   |       $10^{2}$
DMControlEnvWithPhysics('pendulum', 'swingup')        |       0       |       0      |        0
DMControlEnv('point_mass', 'easy')                    |       0       |   $10^{1}$   |       $10^{1}$
DMControlEnvWithPhysics('point_mass', 'easy')         |       0       |       0      |        0
DMControlEnv('reacher', 'easy')                       |    $10^{0}$   |   $10^{3}$   |       $10^{3}$
DMControlEnvWithPhysics('reacher', 'easy')            |    $10^{0}$   |   $10^{0}$   |       $10^{0}$
DMControlEnv('reacher', 'hard')                       |       0       |   $10^{3}$   |       $10^{3}$
DMControlEnvWithPhysics('reacher', 'hard')            |       0       |   $10^{0}$   |       $10^{0}$
DMControlEnv('swimmer', 'swimmer6')                   |    $10^{-1}$  |   $10^{1}$   |       $10^{1}$
DMControlEnvWithPhysics('swimmer', 'swimmer6')        |    $10^{-1}$  |   $10^{0}$   |       $10^{0}$
MControlEnv('swimmer', 'swimmer15')                   |    $10^{0}$   |   $10^{1}$   |       $10^{1}$
DMControlEnvWithPhysics('swimmer', 'swimmer15')       |    $10^{-1}$  |   $10^{0}$   |       $10^{0}$
DMControlEnv('walker', 'stand')                       |       0       |   $10^{-1}$  |       $10^{-1}$
DMControlEnvWithPhysics('walker', 'stand')            |       0       |       0      |        0
DMControlEnv('walker', 'walk')                        |       0       |   $10^{-1}$  |       $10^{-1}$
DMControlEnvWithPhysics('walker', 'walk')             |       0       |       0      |        0
DMControlEnv('walker', 'run')                         |       0       |   $10^{-1}$  |       $10^{-1}$
DMControlEnvWithPhysics('walker', 'run')              |       0       |       0      |        0
GymEnv('Ant-v3')                                      |       0       |   $10^{0}$   |       $10^{0}$
GymEnv('HalfCheetah-v3')                              |       0       |   $10^{-13}$ |       $10^{-14}$
GymEnv('Hopper-v3')                                   |       0       |   $10^{-16}$ |       $10^{-16}$
GymEnv('Humanoid-v3')                                 |       0       |   $10^{1}$   |       $10^{1}$
GymEnv('HumanoidStandup-v2')                          |    $10^{2}$   |   $10^{2}$   |       $10^{-2}$
GymEnv('InvertedDoublePendulum-v2')                   |       0       |       0      |        0
GymEnv('InvertedPendulum-v2')                         |       0       |       0      |        0
GymEnv('Reacher-v2')                                  |    $10^{0}$   |    $10^{-1}$ |       $10^{-1}$
GymEnv('Swimmer-v3')                                  |       0       |    $10^{-15}$|       $10^{-17}$
GymEnv('Walker2d-v3')                                 |       0       |    $10^{-14}$|       $10^{-15}$

# mujoco-py for gym

## Installation for Windows
Support for Windows has been dropped in newer versions of mujoco-py. 
The latest working version is 1.50.1.68.
But even here you can’t do without dancing with a tambourine.

#### Requirements
Microsoft Visual C++ 14.0 or greater.
https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### Installation order
1. Download binaries: http://www.roboti.us/download.html/mjpro150_win64.zip
and activation key: http://www.roboti.us/license.html

2. Create directory ```%userprofile%/.mujoco/```.
3. Unzip the binaries and move the key to the created
    directory.
4. Add full path to directory 
   ```%userprofile%/.mujoco/mjpro150/bin``` into a variable 
   environments PATH.
5. Download mujoco-py versions 1.50.1.68:
https://files.pythonhosted.org/packages/cf/8c/64e0630b3d450244feef0688d90eab2448631e40ba6bdbd90a70b84898e7/mujoco-py-1.50.1.68.tar.gz
6. Unzip the downloaded archive to an arbitrary directory,
    navigate to this directory in the terminal and install
    mujoco-py using command: ```python setup.py install```
   
#### Usage
1. Before each use, you must execute the commands
```python
import os
os.add_dll_directory(os.path.join(os.path.expanduser('~'), ".mujoco", "mjpro150", "bin"))
from mujoco_py import GlfwContext
GlfwContext(True)
```
2. Loading environments is done via [gym](https://www.gymlibrary.ml/environments/mujoco/#)
```python
import gym
env = gym.make("Ant-v3")
```
3. Further use of environments - habitual.

#### Checking work
```python
import gym
import matplotlib.pyplot as plt

env = gym.make("Ant-v3")
state = env.reset()
pixels = env.render("rgb_array")
plt.imshow(pixels)
```

#### List of default mujoco-py environments:
- Ant-v3
- HalfCheetah-v3
- Hopper-v3
- Humanoid-v3
- HumanoidStandup-v2
- InvertedDoublePendulum-v2
- InvertedPendulum-v2
- Pusher-v2
- Reacher-v2
- Swimmer-v3
- Walker2d-v3

## Installation for Linux, OSX

Follow instructions

https://github.com/openai/mujoco-py/blob/master/README.md
