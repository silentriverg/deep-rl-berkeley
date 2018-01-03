from collections import namedtuple
from dqn_utils import *

import gym.spaces
import ipdb
import itertools
import numpy as np
import sys
import tensorflow as tf

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
NUM_ACTION_TIA = 166
NUM_TIMER = 13
EFF_RANGE = 5

def controller(state, setpoint, t):
    # setpoint top bound 38 bottom bound 203
    # action 0,1 not move; 2,4 up; 3,5 down
    setpoint = setpoint + 38

    if state > setpoint + EFF_RANGE and t%2 == 0:
        controller_output = 2
    elif state < setpoint - EFF_RANGE and t%2 == 1:
        controller_output = 3
    else:
        controller_output = 1
    return controller_output

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    #num_actions = env.action_space.n
    num_actions = NUM_ACTION_TIA * NUM_TIMER

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    target_q = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)

    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    target_val = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, axis=1)
    q_val = tf.reduce_sum(tf.one_hot(act_t_ph, num_actions) * q, axis=1)
    total_error = tf.reduce_sum(tf.squared_difference(q_val, target_val))

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        idx = replay_buffer.store_frame(last_obs)
        q_input = replay_buffer.encode_recent_observation()

        choose_random_action = np.random.rand() < exploration.value(t)

        if model_initialized and not choose_random_action:
            action_values = session.run(q, feed_dict={obs_t_ph: [q_input]})
            action = np.argmax(action_values)
            setpoint = int(action / NUM_TIMER)
            timer = action % NUM_TIMER
            cum_reward = 0
        else:
            action = np.random.randint(num_actions)
            setpoint = int(action / NUM_TIMER)
            timer = action % NUM_TIMER
            cum_reward = 0

        #print("setpoint %d" % (setpoint+38))
        #print("timer %d" % (timer))

        for tt in range(timer):
            agent_pos = last_obs[51]
            controller_output = controller(agent_pos, setpoint, tt)
            last_obs, reward, done, info = env.step(controller_output)
            cum_reward = cum_reward + reward

            if done:
                last_obs = env.reset()
                break


        replay_buffer.store_effect(idx, action, cum_reward, done)

        #print (last_obs)
        #print(action)



        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            (
                obs_batch, act_batch, rew_batch, next_obs_batch, done_masks
            ) = replay_buffer.sample(batch_size)

            # 3.b
            if not model_initialized:
                print (80 * "=")
                print ("INITIALIZING THE MODELINO")
                initialize_interdependent_variables(session, tf.global_variables(), {
                    obs_t_ph: obs_batch,
                    obs_tp1_ph: next_obs_batch
                })
                print (80 * "=")
                session.run(update_target_fn)
                model_initialized = True

            # 3.c
            session.run(train_fn, feed_dict={
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                done_mask_ph: done_masks,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })

            # 3.d: periodically update the target network by calling
            if t // target_update_freq > num_param_updates:
                print (80 * "=")
                print ("Updating the target network")
                print (80 * "=")
                session.run(update_target_fn)
                num_param_updates += 1
            """
            else:
                if t % 100 == 0:
                    print "Still no update, t= {}, target_update_freq ={}, t // target_update_freq = {}, num_param_updates = {}".format(t, target_update_freq, t // target_update_freq, num_param_updates)
            """

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            max_action_value = np.max(action_values)
            print("Timestep {:,}".format(t))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("max Q-value %f" % max_action_value)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
