import gym
import pandas as pb
import tensorflow as tf
import time
import dqn

env = gym.make('Pong-ram-v0')
env.reset()
env.unwrapped.get_action_meanings()

initialized = False

for t in range(1000):
    env.render()
    setpoint = 100

    if not initialized:
        control_output = 0
        initialized = True
    else:
        control_output = dqn.controller(last_obs[51], setpoint, t)

    last_obs, reward, done, info = env.step(control_output)
    #last_obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    #last_obs, reward, done, info = env.step(3)
    #print (last_obs)
    #out = tf.concat(last_obs[4:5], last_obs[8:9], last_obs[11:13], last_obs[21:22], last_obs[50:51], last_obs[60:61],last_obs[64:65])
    #print (last_obs[21])
    print (last_obs[51])
    time.sleep(0.2)
    # resolution 210*160
    # 21: vertical value of ball;
    # 49: horizontal value of ball
    # 50: vertical value of ball
    # 51: vertical of agent 203-38
    # 54: vertical of opponent
    # 60: vertical of agent

    # action 0,1 not move 2,4 up 3,5 down
