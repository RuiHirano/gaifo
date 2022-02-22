from gaifo.gaifo import DemoLoader
import pathlib

loader = DemoLoader()
loader.load(pathlib.Path("~/logdir/breakout/train_episodes").expanduser(), capacity=10000)