# -*- coding: utf-8 -*-

import gym
from A3C_Network import A3C_Network
from Utils import process_input
import numpy as np
import os
import tensorflow as tf
from Utils import display_transition
from gym import wrappers
from constants import T

class Worker:
    # each worker has its own network and its own environment
    def __init__(self, args, thread_id, model_path, global_train_counter, global_network, trainer, lock, learning_rate):
        print ("Creating worker: ", thread_id)
        self.args = args
        self.thread_id = thread_id
        self.trainer = trainer
        self.model_path = model_path
        self.global_train_counter = global_train_counter
        self.global_network = global_network
        self.scope = "worker_" + str(thread_id)
        self.state = None
        self.lives = None
        self.lock = lock
        self.T_max = self.args.T_max # total frames to train

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        # with self.lock:
        # creates own worker agent environment
        self._env = gym.make(self.args.game_name)

        if self.thread_id == 0 or self.args.mode == "test":
            #env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id % 10 == 0)

            self._env = wrappers.Monitor(self._env, './results/videos/' + self.args.game_name + "/" + self.args.mode, force=True)
            #self._env = self._env.env

        # get the number of available actions
        self.n_actions = self._env.action_space.n

        # reset openai gym environment and create first state
        self.reset_game_env()
        self.game_initial_setup()

        # creates own worker agent network
        if self.args.mode == "train":
            self.network = A3C_Network(args=args, output_size=self.n_actions, trainer=trainer, scope=self.scope, global_step=self.global_train_counter, learning_rate=learning_rate)

    # for the first step, the state is the same frame screen repeated [agent_history_length] times
    def game_initial_setup(self):
        frame, _, _, _ = self._env.step(0)
        processed_frame = process_input(frame, self.args.screen_width, self.args.screen_height)
        self.state = np.stack(tuple(processed_frame for _ in range(self.args.agent_history_length)), axis=2)
        assert(self.state.shape == (self.args.screen_width, self.args.screen_height, self.args.agent_history_length))
        self._env.frameskip = self.args.frame_skip
        print ("Game setup and ready to go!")

    # Select a random action based on the current policy function probabilities
    def choose_action_randomly(self, action_distributions):
        a = np.random.choice(action_distributions, p=action_distributions)
        action_id = np.argmax(action_distributions == a)
        return action_id

    def reset_game_env(self):
        # with self.lock:
        self._env.reset()
        if self.thread_id == 0:
            self.lives = self._env.env.env.ale.lives()
        else:
            self.lives = self._env.env.ale.lives()

    def process_lives(self):
        terminal = False

        if self.thread_id == 0:
            if self._env.env.env.ale.lives() > self.lives:
                self.lives = self._env.env.env.ale.lives()

            # Loosing a life will trigger a terminal signal in training mode.
            # We assume that a "life" IS an episode during training, but not during testing
            elif self._env.env.env.ale.lives() < self.lives:
                self.lives = self._env.env.env.ale.lives()
                terminal = True
        else:
            if self._env.env.ale.lives() > self.lives:
                self.lives = self._env.env.ale.lives()

            # Loosing a life will trigger a terminal signal in training mode.
            # We assume that a "life" IS an episode during training, but not during testing
            elif self._env.env.ale.lives() < self.lives:
                self.lives = self._env.env.ale.lives()
                terminal = True
        return terminal

    def load_model(self, sess):
        # Restore variables from disk.
        self.saver.restore(sess, self.model_path + "/model.ckpt")
        print("Model restored.")

    def evaluate(self, sess):
        print ("Initializing agent's evaluation.")
        self.load_model(sess)
        total_episode_reward = 0
        episode_number = 0

        for i_episode in range(self.args.T_max):

            if self.args.render:
                self._env.render()

            # Perform action a_t according to policy π(a_t|s_t; θ')
            policy = self.global_network.predict_policy(sess, np.expand_dims(self.state, axis=0))
            action = np.argmax(policy)

            observation, reward, done, info = self._env.step(action)
            total_episode_reward += reward

            # create new state using
            new_observation = np.expand_dims(process_input(observation), axis=2)  # 84 x 84 x 1
            next_state = np.array(self.state[:, :, 1:], copy=True)
            next_state = np.append(next_state, new_observation, axis=2)

            self.state = next_state

            if done:
                self.global_network.update_episode_average_reward(sess, total_episode_reward, episode_number)
                print ("Summary data has been written.")
                total_episode_reward = 0
                episode_number += 1
                print("Episode finished after {} timesteps".format(i_episode + 1))
                self._env.reset()


    def work(self, sess, coordinator):

        with self.lock:
            print ("Thread:", self.thread_id, "has started.")
        global T # global frame counter
        episode_number = 0
        total_episode_reward = 0
        t = 0
        average_episode_reward = []

        while True:
            # print "Time step:", t
            t_start = t

            # reset the worker's local network weights to be the same of the global network
            self.network.sync_local_net(sess)

            if t % 80000 == 0:
                if self.thread_id == 0 and self.args.mode == "train":
                    self.save_model(sess)

            experiences = []

            while True:

                # Perform action a_t according to policy π(a_t|s_t; θ')
                policy, value = self.network.predict_policy_and_values(sess, np.expand_dims(self.state, axis=0))
                action_index = self.choose_action_randomly(policy)

                # with self.lock:
                # self._env.render()
                observation, reward, is_terminal, info = self._env.step(action_index)

                # accumulate immediate reward
                total_episode_reward += reward

                if self.args.mode == "train":
                    has_lost_life = self.process_lives()

                if has_lost_life:
                    reward = -10.0

                # create new state using
                new_observation = np.expand_dims(process_input(observation), axis=2) # 84 x 84 x 1
                next_state = np.array(self.state[:, :, 1:], copy=True)
                next_state = np.append(next_state, new_observation, axis=2)

                # clip the rewards
                clipped_reward = np.clip(reward, self.args.min_reward, self.args.max_reward)

                # store experiences
                ex = [self.state, action_index, clipped_reward, value, is_terminal]
                # if self.thread_id == 0:
                #     display_transition(self._env.get_action_meanings(), ex)

                experiences.append(ex)

                # update local thread counter
                t += 1

                # update global thread counter
                # with self.lock:
                T += 1

                self.state = next_state

                # The Policy and Value functions are updated after every t_max actions or when a terminal state is reached
                if t - t_start == self.args.t_max or is_terminal:
                    break

            R = 0.0
            if not is_terminal:
                R = self.network.predict_values(sess, np.expand_dims(self.state, axis=0))

            self.compute_and_accumulate_rewards(R, experiences, sess)

            if is_terminal:

                average_episode_reward.append(total_episode_reward)
                total_episode_reward = 0
                episode_number += 1

                if episode_number % self.args.average_episode_reward_stats_per_game == 0:
                    if self.thread_id == 0:
                        # print "total_episode_reward", average_episode_reward
                        # print "average episode reward:", np.mean(average_episode_reward)

                        self.network.update_episode_average_reward(sess, np.mean(average_episode_reward), episode_number)
                        print ("Summary data has been written.")

                    average_episode_reward = []

                if self.thread_id == 0:
                    print ("Episode #", episode_number, "has finished. Local step:", t, "Global step:", T)

                # reset environment
                self.reset_game_env()

            if T >= self.T_max:
                break


        self._env.monitor.close()
        print ("Thread:", self.thread_id, "has finished.")

    def compute_and_accumulate_rewards(self, R, experiences, sess):
        previous_states = [d[0] for d in experiences]
        actions = [d[1] for d in experiences]
        rewards = [d[2] for d in experiences]
        values = [d[3] for d in experiences]
        terminals = [d[4] for d in experiences]

        previous_states.reverse()
        actions.reverse()
        rewards.reverse()
        values.reverse()
        terminals.reverse()

        batch_states = []
        batch_actions_one_hot = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for(state, action, reward, value) in zip(previous_states, actions, rewards, values):
            R = reward + self.args.discount_factor * R
            td = R - value # (R - V(si; θ'v)
            a = np.zeros([self.n_actions])
            a[action] = 1

            batch_states.append(state)
            batch_actions_one_hot.append(a)
            batch_td.append(td)
            batch_R.append(R)

        _ = self.network.update_gradients(sess, batch_states, batch_actions_one_hot, batch_td, batch_R, self.thread_id)

    def compare_global_and_local_networks(self, sess):
        current_state = np.expand_dims(self.state, axis=0)

        global_net_policy, global_net_values = self.global_network.predict_policy_and_values(sess, current_state)
        local_nets_policy, local_nets_values = self.network.predict_policy_and_values(sess, current_state)

        if np.array_equal(global_net_policy, local_nets_policy) and np.array_equal(global_net_values, local_nets_values):
            return True
        else:
            return False

    def save_model(self, sess):
        model_path = "./model/" + self.args.game_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print ("Model folder created.")
        save_path = self.saver.save(sess, model_path + "/" + "model.ckpt")
        print("Model saved in file: %s" % save_path)