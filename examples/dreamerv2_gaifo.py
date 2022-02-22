from gaifo import GAIfO, RewardNet, defaults as gaifo_defaults
from dreamerv2.lib import DreamerV2, defaults as dv2_defaults
import gym
import numpy as np
import pathlib

def convert_dv2_episodes(dv2_eps):
	# this is for dreamer episode
	# eps = [ep, ...]
	# ep = {'image': (B, W, H, C), 'reward': (B, 1), 'is_first': (), 'is_last': (), 'is_terminal': (), 'action': ()}
	# tranisitons = {'state': (), 'reward': (), 'done': (), 'next_state': (), 'action': ()}
	transitions = {'state': [], 'next_state': [], 'done': [], 'reward': [], 'action': []}
	for ep in dv2_eps:
		eplen = len(ep['action'])-1
		transitions['state'].extend(ep['image'][:eplen])
		transitions['next_state'].extend(ep['image'][1:])
		transitions['done'].extend(ep['is_terminal'][1:])
		transitions['reward'].extend(ep['reward'][1:])
		transitions['action'].extend(ep['action'][1:])
	return transitions

def convert_episodes_to_transitions(eps):
	# ep = {'state': (), 'reward': (), 'done': (), 'next_state': (), 'action': ()}
	# tranisitons = {'state': (), 'reward': (), 'done': (), 'next_state': (), 'action': ()}
	transitions = {'state': [], 'next_state': [], 'done': [], 'reward': [], 'action': []}
	for ep in eps:
		transitions['state'].extend(ep['state'])
		transitions['next_state'].extend(ep['next_state'])
		transitions['done'].extend(ep['done'])
		transitions['reward'].extend(ep['reward'])
		transitions['action'].extend(ep['action'])
	return transitions

def load_episodes(directory, capacity=None, minlen=1):
	# The returned directory from filenames to episodes is guaranteed to be in
	# temporally sorted order.
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
	episodes = []
	for filename in filenames:
		try:
			with filename.open('rb') as f:
				episode = np.load(f)
				episode = {k: episode[k] for k in episode.keys()}
		except Exception as e:
			print(f'Could not load episode {str(filename)}: {e}')
			continue
		episodes.append(episode)
	return episodes

if __name__ == '__main__':
	# create demo
	demo_dir = pathlib.Path("/Users/ruihirano/MyProjects/FelixPort/data_science/imitaion/expert/pong/demo").resolve()
	episodes = load_episodes(demo_dir, capacity=100000)
	demos = convert_episodes_to_transitions(episodes)
	'''import cv2
	for i in range(10):
		print(np.array(demos['state'][i]).shape)
		cv2.imshow("test", demos['state'][i])
		cv2.waitKey(0)
	import sys
	sys.exit(1)'''
	#print(len(demos['action']), len(episodes))
	# setup gaifo
	reward_net = RewardNet()
	gaifo_config = gaifo_defaults.update({
		'logdir': str(pathlib.Path(__file__).parent.joinpath('logdir/pong_dreamerv2').resolve()),
		'stop': {
			'reward_mean': 0.8,
			'accuracy_mean': 0.9,
			'latest_target_num': 10
		},
		'num_train_epochs': 100,
		'save_weights_iter': 1e3
	}).parse_flags()
	print(gaifo_config)
	gaifo = GAIfO(reward_net=reward_net, config=gaifo_config)

	# setup env
	env = gym.make('Pong-v0')
	env = gym.wrappers.ResizeObservation(env, (64, 64))
	env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
	env = gaifo.update_reward_wrapper(env)

	eval_env = gym.make('Pong-v0')
	eval_env = gym.wrappers.ResizeObservation(eval_env, (64, 64))
	eval_env = gym.wrappers.GrayScaleObservation(eval_env, keep_dim=True)

	# setup dreamerv2
	dv2_config = dv2_defaults.update({
		'logdir': str(pathlib.Path(__file__).parent.joinpath('logdir/pong_dreamerv2').resolve()),
		'log_every': 1e3,
		'prefill': 1e4,
		'grayscale': False,
		'resize': None,
		'steps': 1e8,    # total step
		'rollout_episodes': 1, # rollout episodes during one train epoch
		'train_every': 1e4,
		'train_steps': 100,
	}).parse_flags()
	dv2 = DreamerV2(env, eval_env, dv2_config)

	# add transitions to gaifos' replay
	eps = dv2.get_replay_episodes()
	transitions = convert_dv2_episodes(eps)
	gaifo.add_agent_transitions(transitions)
	gaifo.add_demo_transitions(demos)

	def dv2_train_step_callback(eps):
		rewards = []
		for ep in eps:
			rewards.extend(ep['reward'])
		reward_mean = np.array(rewards).mean()
		print("reward_mean", reward_mean)
		if reward_mean > gaifo_config.stop.reward_mean:
			return True
		return False

	# callback of train gaifo
	def train_gaifo(eps):
		transitions = convert_dv2_episodes(eps)
		gaifo.add_agent_transitions(transitions)
		gaifo.train()

	# pre train gaifo
	gaifo.train()
	# train dreamerv2
	dv2.on_per_train_step(dv2_train_step_callback)
	dv2.on_per_train_epoch(train_gaifo)
	dv2.train()

# TODO: discriminatorの重みを保存する
# TODO: 重みを定期的に別の名前で保存する
# TODO: discのlossがおかしい
	# とりあえずtf.functionをなくした
# 動作検証

# TODO: 報酬を-1~1に
# TODO: dreamerv2にもbreak callbackをつける
