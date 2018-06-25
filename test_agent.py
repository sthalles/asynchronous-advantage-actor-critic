# -*- coding: utf-8 -*-
import pyglet
import gym
import os
import argparse
from A3C_Network import A3C_Network
from Worker import Worker
from constants import GLOBAL_NETWORK_NAME
import tensorflow as tf

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game_name", default="Breakout-v0", help="Atari game name to be used.")
envarg.add_argument("--render", type=bool, default=False, help="Atari game name to be used.")
envarg.add_argument('--mode', choices=['train', 'test'], default='test', help='Mode to run the agent.')
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--agent_history_length", type=int, default=4, help="The number of most recent frames experienced by the agent that are given as input to the Q network.")
envarg.add_argument("--T_max", type=int, default=10000000, help="Total number of steps to train (measured in processed frames)")

mainarg = parser.add_argument_group('Debugging variables')
mainarg.add_argument("--average_episode_reward_stats_per_game", type=int, default=5, help="Show learning statistics after this number of epoch.")
mainarg.add_argument("--update_tf_board", type=int, default=10, help="Update the Tensorboard every X steps.")

args = parser.parse_args()

tf.reset_default_graph()

model_path = os.path.join('./model/', args.game_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# create openai game
env = gym.make(args.game_name)

n_actions = env.action_space.n
global_frame_counter = 0
with tf.Session() as sess:

    with tf.device("/cpu:0"):
        # create global network
        global_network = A3C_Network(args, n_actions, trainer=None, scope=GLOBAL_NETWORK_NAME) # Generate global network

        worker = Worker(args, 0, model_path, None, global_network, None, None, None)

        sess.run(tf.global_variables_initializer())

        worker.evaluate(sess)
