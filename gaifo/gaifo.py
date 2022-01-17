
import torch
from torch._C import TracingState
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import NamedTuple, List
import datetime
import io
import pathlib
import uuid
import gym
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class Config(NamedTuple):
    # learning config
    env_name: str = "CartPole-v0"  # OpenAI gym env id
    demo: str = "CartPole-v0_human.pkl"
    seed: int = 1234  # random seed
    num_updates: int = 10000  # total training iterations
    log_step: int = 100  # log frequency
    play: bool = True  # render after training
    # hyper-parameters
    batch_size: int = 32  # batch size
    num_generator_epochs: int = 3  # number of epochs for training ppo policy
    num_discriminator_epochs: int = 3  # number of epochs for training discriminator
    num_rollout_steps: int = 128
    num_steps: int = 128  # horizon
    num_units: int = 64  # fc units
    gamma: float = 0.99  # discount rate
    lambda_: float = 0.95  # gae discount rate
    clip: float = 0.2  # clipping c
    vf_coef: float = 0.5  # coefficient of value loss
    ent_coef: float = 0.01  # coefficient of entropy
    learning_rate: float = 2.5e-4  # learning rate
    gradient_clip: float = 0.5  # gradinet clipping

class Transition(NamedTuple):
    state: List[List[float]]
    action: int
    reward: float
    next_state: List[List[float]]
    done: bool

class DemoLoader:
    def __init__(self, directory):
        self._directory = directory
        self._demos = self.load(self._directory)

    def _save(self):
        pass

    def _on_step(self):
        self._save()

    def _rollout(self, policy):
        return transitions

    def create(self, policy, steps=100):
        demos = self._rollout(policy, steps)
        self.demos.extend(demos)

    def load(self, directory: str, capacity=None):
        filenames = sorted(directory.glob('*.npz'))
        if capacity:
            num_steps = 0
            num_episodes = 0
            for filename in reversed(filenames):
                length = int(str(filename).split('-')[-1][:-4])
                num_steps += length
                num_episodes += 1
                if num_steps >= capacity:
                    break
            filenames = filenames[-num_episodes:]
        episodes = {}
        for filename in filenames:
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f'Could not load episode {str(filename)}: {e}')
                continue
            episodes[str(filename)] = episode
        return self.demos
    
    def get_demos(self):
        return self._demos

class ReplayMemory:
    def __init__(
        self,
        capacity: int = 0,
        batch_size: int = 0
    ):
        self._capacity = capacity
        self._batch_size = batch_size
        self._transactions = []

    def add_transitions(self, transitions: List[Transition]):
        self._transactions.extend(transitions)

    def sample_transitions(self):
        trs = random.sample(self._transactions, self._batch_size)
        return trs

    def __len__(self):
        return len(self._transactions)

class UpdateRewardWrapper(gym.Wrapper):
    def __init__(self, env, discriminator):
        super().__init__(env)
        self._disc = discriminator

    def reset(self):
        state = self.env.reset()
        self._state = state
        return state
    
    def step(self, action):
        next_state, reward, done, info =  self.env.step(action)
        reward = self._disc.inference(self._state, next_state)
        self._state = next_state
        return next_state, reward, done, info

class GAIfO:
    def __init__(self, reward_net, demo_dir, config):
        self._config = config
        logdir = pathlib.Path(self._config.logdir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        config.save(logdir / 'config.gaifo.yaml')
        self._disc = Discriminator(reward_net)
        self._agent_replay = ReplayMemory(directory=logdir / 'train_episodes', capacity=config.agent_replay_capacity)
        self._demo_replay = ReplayMemory(directory=demo_dir, capacity=config.demo_replay_capacity)

    def add_transitions(self, transitions):
        self._agent_replay.add_transitions(transitions)

    def train(self, num_train_epochs=1):
        # train discriminator
        for _ in num_train_epochs:
            agent_trans = self._agent_replay.sample_transitions()
            demo_trans = self._demo_replay.sample_transitions()
            self._disc.train(agent_trans, demo_trans)

    def update_rewards(self, env):
        env = UpdateRewardWrapper(env, self._disc)
        return env

class RewardNet(tf.keras.Model):
    def __init__(self, state_shape, name="Discriminator"):
        super().__init__(self, name=name)
        self.conv1 = layers.Conv2D(chn=32, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv2 = layers.Conv2D(chn=64, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv3 = layers.Conv2D(chn=64, conv_kernel=(3,3), isPool=False)
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dence1 = layers.Dense(64, activation='relu')
        self.dence2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dence1(x)
        x = self.dence2(x)
        return x

    def compute_reward(self, inputs):
        return tf.math.log(self(inputs) + 1e-8)

class Discriminator(nn.Module):
    def __init__(self, reward_net, optimizer):
        super(Discriminator, self).__init__()
        self._model = reward_net
        self._optimizer = optimizer

    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, **kwargs):
        """
        Train GAIfO
        Args:
            agent_states
            agent_next_states
            expert_states
            expected_next_states
        """
        loss, accuracy, js_divergence = self._train_body(
            agent_states, agent_next_states, expert_states, expert_next_states)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
    
    @tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logits = self._model(tf.concat((expert_states, expert_next_states), axis=1))
                fake_logits = self._model(tf.concat((agent_states, agent_next_states), axis=1))
                loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                         tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
            grads = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(
                zip(grads, self._model.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(
            fake_logits, real_logits)
        return loss, accuracy, js_divergence

    def inference(self, states, next_states):
        """
        Infer Reward with GAIfO
        Args:
            states
            next_states
        Returns:
            tf.Tensor: Reward
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        return self._inference_body(inputs)
    
    @tf.function
    def _inference_body(self, inputs):
        with tf.device(self.device):
            return self._model.compute_reward(inputs)
