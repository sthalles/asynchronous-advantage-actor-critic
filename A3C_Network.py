# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from constants import GLOBAL_NETWORK_NAME

# Asynchronous Advantage Actor-Critic (A3C)
class A3C_Network:
    def __init__(self, args, output_size, trainer, scope, global_step=None, learning_rate=None):
        self.args = args
        self.trainer = trainer
        self.scope_name = scope
        self.global_train_step = global_step
        self.learning_rate = learning_rate
        input_shape = (args.screen_width, args.screen_height, args.agent_history_length) # 84 x 84 x 4
        with tf.variable_scope(scope):
            self.output_size = output_size
            self._input = tf.placeholder(tf.float32,
                                         shape=(None,) + (input_shape[0], input_shape[1], input_shape[2]),
                                         name="input_state")
            self._target = tf.placeholder(tf.float32, [None], name="input_targets")
            self._action = tf.placeholder(tf.float32, [None, output_size], name="input_actions_one_hot")
            self._build_graph()

            if scope != GLOBAL_NETWORK_NAME:
                # create sync and losses operations
                self.sync = self._prepare_sync_ops()
                self._prepare_loss_ops()

            summary_path = './summary/' + self.args.game_name + "/" + self.args.mode + "/"
            self.train_writer = tf.summary.FileWriter(summary_path)

            self.episode_average_reward_input = tf.placeholder(tf.float32, [], name="average_score_per_episode_place_holder")
            self.episode_average_reward_summary = tf.summary.scalar('average_score_per_episode', self.episode_average_reward_input)

    def update_episode_average_reward(self, sess, average_score, game_number):
        reward_summary = sess.run(self.episode_average_reward_summary,
                                  feed_dict={self.episode_average_reward_input: average_score})
        self.train_writer.add_summary(reward_summary, game_number)



    # Build network as described in (Mnih et al., 2013)
    def _build_graph(self):

        normalized_input = tf.div(self._input, 255.0)

        #d = tf.divide(1.0, tf.sqrt(8. * 8. * 4.))
        conv1 = slim.conv2d(normalized_input, 16, [8, 8], activation_fn=tf.nn.relu,
                            padding='VALID', stride=4, biases_initializer=None)
                            # weights_initializer=tf.random_uniform_initializer(minval=-d, maxval=d))

        #d = tf.divide(1.0, tf.sqrt(4. * 4. * 16.))
        conv2 = slim.conv2d(conv1, 32, [4, 4], activation_fn=tf.nn.relu,
                            padding='VALID', stride=2, biases_initializer=None)
                            #weights_initializer=tf.random_uniform_initializer(minval=-d, maxval=d))

        flattened = slim.flatten(conv2)

        #d = tf.divide(1.0, tf.sqrt(2592.))
        fc1 = slim.fully_connected(flattened, 256, activation_fn=tf.nn.relu, biases_initializer=None)
                                   #weights_initializer=tf.random_uniform_initializer(minval=-d, maxval=d))

        #d = tf.divide(1.0, tf.sqrt(256.))
        # estimate of the value function
        self.value_func_prediction = slim.fully_connected(fc1, 1, activation_fn=None, biases_initializer=None)
                                                          #weights_initializer=tf.random_uniform_initializer(minval=-d, maxval=d))

        # softmax output with one entry per action representing the probability of taking an action
        self.policy_predictions = slim.fully_connected(fc1, self.output_size, activation_fn=tf.nn.softmax,
                                                       biases_initializer=None)
                                                       #weights_initializer=tf.random_uniform_initializer(minval=-d, maxval=d))

    def predict_values(self, sess, states):
        return sess.run(self.value_func_prediction, {self._input: states})[0][0]

    def predict_policy(self, sess, states):
        return sess.run(self.policy_predictions, {self._input: states})[0]

    def predict_policy_and_values(self, sess, states):
        policy, values = sess.run([self.policy_predictions, self.value_func_prediction], {self._input: states})
        return policy[0], values[0][0]

    def _prepare_loss_ops(self):
        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None], name="r-v_values")

        # policy softmax probabilities
        policy = self.policy_predictions

        # R (input for value)
        self.r = tf.placeholder("float", [None], name="R_values")

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(policy, 1e-20, 1.0)) #log π(a_i|s_i; θ?)

        # Add print operation
        # log_pi = tf.Print(log_pi, [log_pi], message="log_pi: ")

        # policy entropy
        entropy = - tf.reduce_sum(policy * log_pi, reduction_indices=1)

        # entropy = tf.Print(entropy, [entropy], message="entropy: ")

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
        policy_loss = - tf.reduce_sum(tf.reduce_sum(log_pi * self._action, reduction_indices=1) * self.td + entropy * self.args.entropy_regularization)

        # value loss function (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * tf.nn.l2_loss(self.r - self.value_func_prediction)

        bs = tf.to_float(tf.shape(self._input)[0])

        # gradients of policy and value are summed up
        self.total_loss = value_loss + policy_loss

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
        self.gradients = tf.gradients(self.total_loss, local_vars)

        # get variables from the global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NETWORK_NAME)
        grads, _ = tf.clip_by_global_norm(self.gradients, 40.0)

        # apply the gradients
        self.apply_gradients = self.trainer.apply_gradients(zip(grads, global_vars), global_step=self.global_train_step)

        self.merged = tf.summary.merge([
            tf.summary.scalar('loss', self.total_loss / bs),
            tf.summary.scalar("value_loss", value_loss / bs),
            tf.summary.scalar('policy_loss', policy_loss / bs),
            tf.summary.scalar('learning_rate', self.learning_rate)
        ])

    def update_gradients(self, sess, batch_states, batch_actions_one_hot, batch_td, batch_R, thread_id):
        summaries, grads, train_step, lr = sess.run([self.merged, self.apply_gradients, self.global_train_step, self.learning_rate],
                                                    feed_dict = {
            self._input: batch_states,
            self._action: batch_actions_one_hot,
            self.td: batch_td,
            self.r: batch_R,
        })
        # print "Thread id: ", thread_id, " Learning rate: ", lr
        if train_step % self.args.update_tf_board == 0 and thread_id == 0:
            self.train_writer.add_summary(summaries, train_step)
            # print "Global episode counter:", counter

        return train_step

    def _prepare_sync_ops(self):
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NETWORK_NAME)
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)

        sync_ops = []

        for(src_var, dst_var) in zip(global_vars, local_vars):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def sync_local_net(self, sess):
        sess.run(self.sync)
        # print "Local network:", self.scope_name, "successfully updaded!"