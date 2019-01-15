import os
from time import time
from datetime import datetime
from agent.player import GamePlayer
from config import Config
from env.game_env import GameEnv, Winner, Player
from lib import tf_util
from lib.data_helper import get_game_data_filenames, write_game_data_to_file
from lib.model_helpler import load_best_model_weight, save_as_best_model#, reload_best_model_weight_if_changed

def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    return SelfPlayWorker(config, env=GameEnv()).start()

class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        self.config = config
        self.model = model
        self.env = env     # type: GameEnv
        self.black = None  # type: GamePlayer
        self.white = None  # type: GamePlayer
        self.buffer = []

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        self.buffer = []
        idx = 1

        while True:
            start_time = time()            
            env = self.start_game(idx)
            end_time = time()
            print("game ",idx," time=",(end_time - start_time)," sec, turn=",env.turn," ", env.observation, env.winner)
            idx += 1

    def start_game(self, idx):
        self.env.reset()
        self.black = GamePlayer(self.config, self.model)
        self.white = GamePlayer(self.config, self.model)
        while not self.env.done:
            if self.env.player_turn() == Player.black:
                action = self.black.action(self.env.board)
            else:
                action = self.white.action(self.env.board)
            #self.env.render()
            #print('action to take:',action)
            self.env.step(action)
            #self.env.render()
            #print('\n\n')
            
        self.finish_game()
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()
        return self.env

    def save_play_data(self, write=True):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        print("save play data to ",path)
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0
        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def load_model(self):
        from agent.model import GameModel
        model = GameModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model
