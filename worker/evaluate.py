import os
from random import random

from agent.model import GameModel
from agent.player import GamePlayer
from config import Config
from env.game_env import GameEnv, Winner, Player
from lib import tf_util
from lib.data_helper import get_next_generation_model_dirs
from lib.model_helpler import save_as_best_model, load_best_model_weight
from time import sleep

def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    return EvaluateWorker(config).start()

class EvaluateWorker:
    def __init__(self, config: Config):
        self.config = config
        self.best_model = None

    def start(self):
        self.best_model = self.load_best_model()

        while True:
            ng_model, model_dir = self.load_next_generation_model()
            print("start evaluate model", model_dir)
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                print("New Model become best model:", model_dir)
                save_as_best_model(ng_model)
                self.best_model = ng_model
            self.remove_model(model_dir)

    def evaluate_model(self, ng_model):
        results = []
        winning_rate = 0
        for game_idx in range(self.config.eval.game_num):
            # ng_win := if ng_model win -> 1, lose -> 0, draw -> None
            ng_win, white_is_best = self.play_game(self.best_model, ng_model)
            if ng_win is not None:
                results.append(ng_win)
                winning_rate = sum(results) / len(results)
            print("game", game_idx,": ng_win=",ng_win," white_is_best_model=",white_is_best,"winning rate ",winning_rate)
            if results.count(0) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                print("lose count reach", results.count(0)," so give up challenge")
                break
            if results.count(1) >= self.config.eval.game_num * self.config.eval.replace_rate:
                print("win count reach", results.count(1)," so change best model")
                break

        winning_rate = sum(results) / len(results)
        print("winning rate", winning_rate*100,"%")
        return winning_rate >= self.config.eval.replace_rate

    def play_game(self, best_model, ng_model):
        env = GameEnv().reset()

        best_player = GamePlayer(self.config, best_model, play_config=self.config.eval.play_config)
        ng_player = GamePlayer(self.config, ng_model, play_config=self.config.eval.play_config)
        best_is_white = random() < 0.5
        if not best_is_white:
            black, white = best_player, ng_player
        else:
            black, white = ng_player, best_player

        env.reset()
        while not env.done:
            if env.player_turn() == Player.black:
                action = black.action(env.board)
            else:
                action = white.action(env.board)
            env.step(action)

        ng_win = None
        if env.winner == Winner.white:
            if best_is_white:
                ng_win = 0
            else:
                ng_win = 1
        elif env.winner == Winner.black:
            if best_is_white:
                ng_win = 1
            else:
                ng_win = 0
        return ng_win, best_is_white

    def load_best_model(self):
        model = GameModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            print("There is no next generation model to evaluate")
            sleep(600)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = GameModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)
