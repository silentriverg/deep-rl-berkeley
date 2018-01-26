import gym
import pandas as pb
import tensorflow as tf
import time
import dqn_timer
import numpy as np

MULTIPLE_FRAMES = 4
Num_training_sets = 4000
EFF_RANGE = 5

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

root_path = '/home/zmart/deep-rl-berkeley/hw3/'
tfrecords_filename = root_path + 'tfrecord/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

def generate_train_data():
    env = gym.make('Pong-ram-v0')

    for i in range(Num_training_sets):
        initialization = np.random.randint(38, 202)
        last_obs_mult = np.zeros(128 * MULTIPLE_FRAMES)
        last_obs = env.reset()
        for t in range(50):
            control_output = dqn_timer.controller(last_obs[51], initialization, t)
            last_obs, _, _, _ = env.step(control_output)
            agent_pos = last_obs[51]
            if agent_pos <= initialization + EFF_RANGE and agent_pos >= initialization - EFF_RANGE:
                break
        for t in range(4):
            action = np.random.randint(6)
            last_obs, _, _, _ = env.step(action)
            last_obs_mult = dqn_timer.update_obs(last_obs_mult, last_obs)

        setpoint = np.random.randint(38, 202)
        for t in range(100):
            control_output = dqn_timer.controller(last_obs[51], setpoint, t)
            last_obs, _, _, _ = env.step(control_output)
            agent_pos = last_obs[51]
            if agent_pos <= setpoint + EFF_RANGE and agent_pos >= setpoint - EFF_RANGE:
                counter = t
                break

        last_obs_mult = last_obs_mult.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'state': _bytes_feature(last_obs_mult),
            'setpoint': _int64_feature(setpoint),
            'time': _int64_feature(counter)}))

        writer.write(example.SerializeToString())
        if i%200 == 0:
            print(i)

    writer.close()

def main():
    # Run training
    timer = generate_train_data()
    print(timer)

if __name__ == "__main__":
    main()

