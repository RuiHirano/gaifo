import numpy as np
import pathlib
import uuid
import gym
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from .config import Config
import yaml
from operator import itemgetter
from collections import deque
import pickle
configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = Config(configs.pop('defaults'))

class ReplayMemory:
    def __init__(
        self,
        capacity: int = 0,
        batch_size: int = 32
    ):
        self._capacity = capacity
        self._batch_size = batch_size
        self._transitions = {'state': [], 'next_state': [], 'done': [], 'reward': [], 'action': []}

    def add_transitions(self, transitions):
        def add(key, transitions):
            self._transitions[key].extend(transitions[key])
            # remove items over capacity
            if len(self._transitions[key]) > self._capacity:
                diff = len(self._transitions[key]) - self._capacity
                del self._transitions[key][:diff]
        [add(k, transitions) for k in transitions]

    def sample_transitions(self):
        data_size = len(self._transitions['action'])
        shuffled_idx = np.random.choice(np.arange(data_size), self._batch_size, replace=False)
        sampled_transitions = {
            'state': list(itemgetter(*shuffled_idx)(self._transitions['state'])), 
            'next_state': list(itemgetter(*shuffled_idx)(self._transitions['next_state'])), 
            'done': list(itemgetter(*shuffled_idx)(self._transitions['reward'])), 
            'reward': list(itemgetter(*shuffled_idx)(self._transitions['done'])), 
            'action': list(itemgetter(*shuffled_idx)(self._transitions['action']))
        }
        return sampled_transitions

    def __len__(self):
        return len(self._transitions['action'])


def grayscale(obs, keep_dim=False):
    import cv2
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    if keep_dim:
        obs = np.expand_dims(obs, -1)
    return obs

class UpdateRewardWrapper(gym.Wrapper):
    def __init__(self, env, discriminator):
        super().__init__(env)
        self._disc = discriminator
        self.count = 0

    def reset(self):
        state = self.env.reset()
        #self._state = grayscale(state, keep_dim=True)
        self._state = state
        return state
    
    def step(self, action):
        next_state, reward, done, info =  self.env.step(action)
        reward = self._disc.inference(self._state, next_state)[0][0].numpy()
        #print("reward", reward)
        self.count += 1
        if self.count % 100 == 0:
            print(self.count,": ", reward)
            self.count = 0
        self._state = next_state
        return next_state, reward, done, info

class GAIfO:
    def __init__(self, reward_net, config):
        self._config = config
        self._logdir = pathlib.Path(self._config.logdir).expanduser()
        self._logdir.mkdir(parents=True, exist_ok=True)
        config.save(self._logdir / 'config.gaifo.yaml')
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        if (self._logdir / 'gaifo_weights.hdf5').exists():
            print("load {}".format(str(self._logdir / 'gaifo_weights.hdf5')))
            reward_net.load_weights(self._logdir / "gaifo_weights.hdf5")
        self._disc = Discriminator(reward_net, optimizer)
        self._agent_replay = ReplayMemory(capacity=config.agent_replay_capacity, batch_size=config.batch_size)
        self._demo_replay = ReplayMemory(capacity=config.demo_replay_capacity, batch_size=config.batch_size)
        #self._demo_replay.add_transitions(demos)
        self._num_train_epoch = 0

    def add_agent_transitions(self, transitions):
        print("[Add agent transitions] length: {}, obs shape: {}".format(len(transitions["action"]), np.array(transitions["state"][0]).shape))
        self._agent_replay.add_transitions(transitions)

    def add_demo_transitions(self, transitions):
        print("[Add demo transitions] length: {}, obs shape: {}".format(len(transitions["action"]), np.array(transitions["state"][0]).shape))
        self._demo_replay.add_transitions(transitions)

    def train(self):
        # train discriminator
        latest_accs = np.zeros(10)
        for _ in range(self._config.num_train_epochs):
            agent_trans = self._agent_replay.sample_transitions()
            demo_trans = self._demo_replay.sample_transitions()
            loss, accuracy, js_divergence = self._disc.train(agent_trans['state'], agent_trans['next_state'], demo_trans['state'], demo_trans['next_state'])
            self._num_train_epoch += 1
            if self._num_train_epoch % self._config.save_weights_iter == self._config.save_weights_iter-1:
                self._disc.save(self._logdir / 'gaifo_weights.hdf5')
            #latest_accs.append(accuracy)
            latest_accs[0:-1] = latest_accs[1:]
            latest_accs[-1] = accuracy
            mean_accuracy = latest_accs.mean()
            if mean_accuracy > self._config.stop.accuracy_mean:
                print("[stop train epoch]: mean_accuracy: {}".format(mean_accuracy))
                break


    def update_reward_wrapper(self, env):
        env = UpdateRewardWrapper(env, self._disc)
        return env

class RewardNet(tf.keras.Model):
    def __init__(self, name="Discriminator"):
        super().__init__(self, name=name)
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", strides=1)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", strides=1)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dence1 = layers.Dense(64, activation='relu')
        self.dence2 = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        # x: (B, W, H, C)
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
        #reward = tf.math.log(self(inputs) + 1e-8)
        reward = self(inputs)
        return reward

class Discriminator(tf.Module):
    def __init__(self, reward_net, optimizer, gpu=0):
        super(Discriminator, self).__init__()
        self._model = reward_net
        self._optimizer = optimizer
        self._num_train = 0
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

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
        self._num_train += 1
        print("[{}] loss: {}, accuracy: {}, js_div: {}".format(self._num_train, loss, accuracy, js_divergence))
        return loss, accuracy, js_divergence
        #tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        #tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        #tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
    
    #@tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                expert_logits = self._model(tf.concat((expert_states, expert_next_states), axis=3)/255, training=True)
                agent_logits = self._model(tf.concat((agent_states, agent_next_states), axis=3)/255, training=True)
                loss = -(tf.reduce_mean(tf.math.log(expert_logits + epsilon)) +
                         tf.reduce_mean(tf.math.log(1. - agent_logits + epsilon)))
            #print("variables", loss, expert_logits, agent_logits)
            grads = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(
                zip(grads, self._model.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(expert_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(agent_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(
            agent_logits, expert_logits)
        return loss, accuracy, js_divergence

    def _compute_js_divergence(self, agent_logits, expert_logits):
        m = (agent_logits + expert_logits) / 2.
        return tf.reduce_mean((
            agent_logits * tf.math.log(agent_logits / m + 1e-8) + expert_logits * tf.math.log(expert_logits / m + 1e-8)) / 2.)

    def convert_states(self, states, next_states):
        # states : (W, H, C), next_states : (W, H, C)
        inputs = np.concatenate((states, next_states), axis=2) # (W, H, 2C)
        inputs = np.expand_dims(inputs, axis=0) # (B, W, H, 2C)
        inputs = inputs/255 # resize value from 0 to 1

    def inference(self, states, next_states):
        """
        Infer Reward with GAIfO
        Args:
            states : (W, H, C)
            next_states : (W, H, C)
        Returns:
            tf.Tensor: Reward
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=2) # (W, H, 2C)
        inputs = np.expand_dims(inputs, axis=0) # (B, W, H, 2C)
        inputs = inputs/255 # resize value from 0 to 1
        reward = self._inference_body(inputs) # (B, 1)
        return reward
    
    @tf.function
    def _inference_body(self, inputs):
        with tf.device(self.device):
            return self._model.compute_reward(inputs)

    def save(self, filename):
        self._model.save_weights(filename, save_format="h5")
        '''values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        print(f'Save checkpoint with {amount} tensors and {count} parameters.')
        with pathlib.Path(filename).open('wb') as f:
            pickle.dump(values, f)'''

    def load(self, filename):
        self._model.load_weights(filename)
        '''with pathlib.Path(filename).open('rb') as f:
            values = pickle.load(f)
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        print(f'Load checkpoint with {amount} tensors and {count} parameters.')
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)'''

