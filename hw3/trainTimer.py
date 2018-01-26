import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

root_path = '/home/zmart/deep-rl-berkeley/hw3/'
tfrecord_filename = root_path + 'tfrecord/train.tfrecords'
checkpoint_filename = root_path + 'model/timer_model.ckpt'

MULTIPLE_FRAMES = 4

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'state':tf.FixedLenFeature([],tf.string),
            'setpoint':tf.FixedLenFeature([],tf.int64),
            'time': tf.FixedLenFeature([], tf.int64)})

    state = tf.decode_raw(features['state'],tf.float64)
    state = tf.reshape(state,[1,128*MULTIPLE_FRAMES])
    setpoint =features['setpoint']
    setpoint = tf.cast(setpoint, tf.float64)
    setpoint = tf.reshape(setpoint,[1,1])
    time = features['time']
    time = tf.cast(time, tf.float64)

    state, setpoint, time = tf.train.batch([state, setpoint, time],
                                           batch_size=4,
                                           capacity=2000)

    return state, setpoint, time

def model(state,setpoint):
    input = tf.concat([state,setpoint],2)
    out = layers.fully_connected(input, num_outputs=1024, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=1, activation_fn=None)

    return out

def train(tfrecord_filename):
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    state, setpoint, time = read_and_decode(filename_queue)

    out = model(state,setpoint)

    loss = tf.reduce_mean(tf.square(out-time))
    optimizer = tf.train.GradientDescentOptimizer(0.2)

    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(10000):
            sess.run(train)
            print(i)

        saver.save(sess,checkpoint_filename)

        coord.request_stop()
        coord.join(threads)



def main():
    # Run training
    train(tfrecord_filename)

if __name__ == "__main__":
    main()