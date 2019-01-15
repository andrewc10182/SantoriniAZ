import os
import sys
import win_unicode_console
from dotenv import load_dotenv, find_dotenv

win_unicode_console.enable()
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if find_dotenv():
    load_dotenv(find_dotenv())

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

import argparse
from config import Config

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=['self', 'opt', 'eval', 'play_gui'])
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="normal")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    return parser

def setup(config: Config, args):
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()

parser = create_parser()
args = parser.parse_args()
config_type = args.type

config = Config(config_type=config_type)
setup(config, args)

print("config type: ", config_type)

if args.cmd == "self":
    from worker import self_play
    self_play.start(config)
elif args.cmd == 'opt':
    from worker import optimize
    optimize.start(config)
elif args.cmd == 'eval':
    from worker import evaluate
    evaluate.start(config)
elif args.cmd == 'play_gui':
    from play_game import gui
    gui.start(config)
