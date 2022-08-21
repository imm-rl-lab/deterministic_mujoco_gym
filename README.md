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
DMControlEnv('ball_in_cup', 'catch')                  |       0       |       0      |      $10^{-3}$
DMControlEnvWithPhysics('ball_in_cup', 'catch')       |       0       |       0      |        0
DMControlEnv('cartpole', 'balance')                   |       0       |       0      |        0
DMControlEnv('cartpole', 'balance_sparse')            |       0       |       0      |        0
DMControlEnv('cartpole', 'swingup')                   |       0       |       0      |        0
DMControlEnv('cartpole', 'swingup_sparse')            |       0       |       0      |        0
DMControlEnv('cheetah', 'run')                        |       0       |       0      |       $10^{-2}$
DMControlEnvWithPhysics('cheetah', 'run')             |       0       |       0      |        0
DMControlEnv('finger', 'spin')                        |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('finger', 'spin')             |       0       |       0      |        0
DMControlEnv('finger', 'turn_easy')                   |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('finger', 'turn_easy')        |       0       |       0      |        0
DMControlEnv('finger', 'turn_hard')                   |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('finger', 'turn_hard')        |       0       |       0      |        0
DMControlEnv('fish', 'upright')                       |       0       |       0      |       $10^{-5}$
DMControlEnvWithPhysics('fish', 'upright')            |       0       |       0      |        0
DMControlEnv('fish', 'swim')                          |   $10^{-5}$   |   $10^{-5}$  |       $10^{-5}$
DDMControlEnvWithPhysics('fish', 'swim')              |   $10^{-5}$   |   $10^{-5}$  |       $10^{-5}$
DMControlEnv('hopper', 'stand')                       |       0       |       0      |       $10^{-9}$
DMControlEnvWithPhysics('hopper', 'stand')            |       0       |       0      |        0
DMControlEnv('hopper', 'hop')                         |       0       |       0      |       $10^{-9}$
DMControlEnvWithPhysics('hopper', 'hop')              |       0       |       0      |        0
DMControlEnv('humanoid', 'stand')                     |       0       |       0      |       $10^{-2}$
DMControlEnvWithPhysics('humanoid', 'stand')          |       0       |       0      |        0
DMControlEnv('humanoid', 'walk')                      |       0       |       0      |       $10^{-2}$
DMControlEnvWithPhysics('humanoid', 'walk')           |       0       |       0      |        0
DMControlEnv('humanoid', 'run')                       |       0       |       0      |       $10^{-2}$
DMControlEnvWithPhysics('humanoid', 'run')            |       0       |       0      |        0
DMControlEnv('manipulator', 'bring_ball')             |   $10^{-59}$  |   $10^{-29}$ |       $10^{-1}$
DMControlEnvWithPhysics('manipulator', 'bring_ball')  |   $10^{-59}$  |   $10^{-29}$ |       $10^{-29}$
DMControlEnv('pendulum', 'swingup')                   |       0       |       0      |       $10^{0}$
DMControlEnvWithPhysics('pendulum', 'swingup')        |       0       |       0      |        0
DMControlEnv('point_mass', 'easy')                    |       0       |       0      |       $10^{-1}$
DMControlEnvWithPhysics('point_mass', 'easy')         |       0       |       0      |        0
DMControlEnv('reacher', 'easy')                       |    $10^{0}$   |       0      |       $10^{-10}$
DMControlEnvWithPhysics('reacher', 'easy')            |    $10^{0}$   |       0      |        0
DMControlEnv('reacher', 'hard')                       |       0       |       0      |       $10^{-10}$
DMControlEnvWithPhysics('reacher', 'hard')            |       0       |       0      |        0
DMControlEnv('swimmer', 'swimmer6')                   |    $10^{-2}$  |   $10^{-3}$  |       $10^{-3}$
DMControlEnvWithPhysics('swimmer', 'swimmer6')        |    $10^{-2}$  |   $10^{-3}$  |       $10^{-3}$
MControlEnv('swimmer', 'swimmer15')                   |    $10^{-3}$  |   $10^{-3}$  |       $10^{-3}$
DMControlEnvWithPhysics('swimmer', 'swimmer15')       |    $10^{-3}$  |   $10^{-3}$  |       $10^{-3}$
DMControlEnv('walker', 'stand')                       |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('walker', 'stand')            |       0       |       0      |        0
DMControlEnv('walker', 'walk')                        |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('walker', 'walk')             |       0       |       0      |        0
DMControlEnv('walker', 'run')                         |       0       |       0      |       $10^{-15}$
DMControlEnvWithPhysics('walker', 'run')              |       0       |       0      |        0
GymEnv('Ant-v3')                                      |       0       |       0      |       $10^{-14}$
GymEnv('HalfCheetah-v3')                              |       0       |       0      |       $10^{-14}$
GymEnv('Hopper-v3')                                   |       0       |       0      |       $10^{-16}$
GymEnv('Humanoid-v3')                                 |       0       |       0      |       $10^{1}$
GymEnv('HumanoidStandup-v2')                          |    $10^{2}$   |    $10^{3}$  |       $10^{-4}$
GymEnv('InvertedDoublePendulum-v2')                   |       0       |       0      |        0
GymEnv('InvertedPendulum-v2')                         |       0       |       0      |        0
GymEnv('Reacher-v2')                                  |    $10^{0}$   |    $10^{-1}$ |       $10^{-1}$
GymEnv('Swimmer-v3')                                  |       0       |       0      |       $10^{-16}$
GymEnv('Walker2d-v3')                                 |       0       |       0      |       $10^{-14}$

*TODO: installation*
