import gym
import numpy as np
import tensorflow as tf
epsilon = 1e-8
print(tf.math.log(1. -1. +epsilon))

shuffled_idx = np.random.choice(np.arange(100), 30, replace=False)
print(shuffled_idx)
transitions = {
    'state': np.arange(100), 
    'next_state': np.arange(100), 
    'done': np.arange(100), 
    'reward': np.arange(100), 
    'action': np.arange(100)
}
from operator import itemgetter
sampled_transitions = {
            'state': list(itemgetter(*shuffled_idx)(transitions['action'])), 
            'next_state': list(itemgetter(*shuffled_idx)(transitions['state'])), 
            'done': list(itemgetter(*shuffled_idx)(transitions['reward'])), 
            'reward': list(itemgetter(*shuffled_idx)(transitions['done'])), 
            'action': list(itemgetter(*shuffled_idx)(transitions['done']))
        }
print(sampled_transitions)