# Mujoco Gym

In this repository, we are trying different ways to make reinforcement learning environments from [Mujoco Gym](https://www.gymlibrary.ml/environments/mujoco/) and [dm_control](https://github.com/deepmind/dm_control) deterministic. We strive to ensure that the environments have the following important properties:

- the function <code>reset()</code> gives always the same initial state;
- the function <code>step(action)</code> such that a sequence of <code>actions</code> uniquely determines a trajectory (<code>states</code>) and <code>rewards</code>;
- the function <code>virtual_step(state, action)</code> uniquely determines <code>next_state</code> and <code>reward</code>.

In many environments in Mujoco, <code>next_state</code> and <code>reward</code> depend not only on the current <code>states</code>, but also on "internal physics". Therefore, the above points are not always fulfilled.
