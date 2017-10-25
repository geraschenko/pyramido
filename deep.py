import argparse
import tensorflow as tf
import random
import sys

from search import MCTS
from pyramido import possible_moves, PyramidoPosition, toArray, leaf_node_value

FLAGS = None

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fully_connected_layer(input_layer, input_size, output_size, index, keep_prob=None):
    with tf.name_scope('fully_connected%d' % index):
        W = weight_variable([input_size, output_size])
        b = bias_variable([output_size])
        layer = tf.nn.relu(tf.matmul(input_layer, W) + b)
        if keep_prob is None:
            return layer
        return tf.nn.dropout(layer, keep_prob)

def pyramido_net(layer):
    keep_prob = tf.placeholder(tf.float32)

    layer = tf.reshape(layer, [tf.shape(layer)[0], 2*31])
    layer = fully_connected_layer(layer, 2*31, 40, 1, keep_prob)
    layer = fully_connected_layer(layer, 40, 40, 2, keep_prob)
    layer = fully_connected_layer(layer, 40, 40, 3, keep_prob)
    layer = fully_connected_layer(layer, 40, 40, 4, keep_prob)
    layer = fully_connected_layer(layer, 40, 1, 5)

    return layer, keep_prob


def nn_policy(y, x, keep_prob):
    def policy(position):
        moves = possible_moves(position)
        if not moves:
            return {}, leaf_node_value(position)
        moves_nn_format = [m.nn_format() for m in moves]
        scores = y.eval(feed_dict={x: moves_nn_format, keep_prob: 1.0}).flatten()
        probability = {}
        for i, m in enumerate(moves):
            probability[m] = scores[i]
        return probability, max(probability.values())
    return policy


def generate_data(policy, num_games, save_location):
    batch = []
    for _ in range(num_games):
        batch.extend(generate_one_game(policy))
    random.shuffle(batch)
    #TODO: make sure those games get saved
    positions, values = zip(*batch)
    positions = [p.nn_format() for p in positions]
    return positions, values


def generate_one_game(policy, num_iterations=100):
    x = []
    y_ = []
    position = PyramidoPosition(toArray(1))
    movenumber = 1
    player = 1
    while possible_moves(position):
        dist, _ = MCTS(policy, position, num_iterations)
        max_probability = max(dist.values())
        for move, probability in dist.items():
            x.append(move)
            y_.append(probability)
            if probability == max_probability:
                position = move
                print('Move %d: Player %d moves with value %f to\n%s'
                        % (movenumber, player, probability, str(position)))
                movenumber, player = movenumber+1, 3-player
    return zip(x, y_)


def main(_):
    # Input board. For each square, two floats to indicate which player piece
    # is there ... the only valid input values are (0, 0), (1, 0), and (0, 1).
    x = tf.placeholder(tf.float32, [None, 2, 31])

    y_ = tf.placeholder(tf.float32, [None, 1])
    y, keep_prob = pyramido_net(x)

    with tf.name_scope('loss'):
        loss = tf.squared_difference(y, y_)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope('mean_loss'):
        mean_loss = tf.reduce_mean(loss)

    # graph_location = 'checkpoints'
    # print('Saving graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver.restore(sess, 'some-model-name')
        sess.run(tf.global_variables_initializer())
        for step in range(100):
            policy = nn_policy(y, x, keep_prob)
            positions, values = generate_data(policy, 100, 'step%d_training' % step)
            for i in range(10):  # number of times to cycle through the data
                batch_size = 100
                start = 0
                while start < len(positions):
                    end = min(start + batch_size, len(positions))
                    batch_positions = positions[start:end]
                    batch_values = values[start:end]
                    start = end
                    train_step.run(feed_dict={
                        x: batch_positions, y_: batch_values, keep_prob: 0.5})
            saver.save(sess, 'model', global_step=step)
            train_loss = loss.eval(feed_dict={
                x: batch_positions, y_: batch_values, keep_prob: 1.0})
            print('step %d, training loss %g' % (step, train_loss))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/pyramido/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
