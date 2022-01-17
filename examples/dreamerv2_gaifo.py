from pyimitation import GAIfO, DemoLoader, RewardNet
from dreamerv2.lib import DreamerV2, defaults
import gym

def convert_episodes(dv2_eps):
	return []

if __name__ == '__main__':
	# create demo
	demo_loader = DemoLoader("./demo")
	demo_loader.create(policy, steps=1000)
	demos = demo_loader.get_demos()

	# setup gaifo
	reward_net = RewardNet()
	gaifo_config = defaults.update({
		'logdir': '~/logdir/breakout',
		'log_every': 1e3,
		'train_every': 10,
		'prefill': 1e5,
		'actor_ent': 3e-3,
		'loss_scales.kl': 1.0,
		'discount': 0.99,
	}).parse_flags()
	gaifo = GAIfO(reward_net=reward_net, demonstrations=demos, config=gaifo_config)
	# setup env
	env = gym.make('Breakout-v0')
	env = gaifo.update_reward_wrapper(env)
	# setup dreamerv2
	dv2_config = defaults.update({
		'logdir': '~/logdir/breakout',
		'log_every': 1e3,
		'train_every': 10,
		'prefill': 1e5,
		'actor_ent': 3e-3,
		'loss_scales.kl': 1.0,
		'discount': 0.99,
	}).parse_flags()
	dv2 = DreamerV2(env, dv2_config)

	# add transitions to gaifos' replay
	eps = dv2.get_replay_episodes()
	transitions = convert_episodes(eps)
	gaifo.add_transitions(transitions)

	# callback of train gaifo
	def train_gaifo(eps):
		transitions = convert_episodes(eps)
		gaifo.add_transitions(transitions)
		gaifo.train()

	# train dreamerv2
	dv2.on_train_epoch(train_gaifo)
	dv2.train()
