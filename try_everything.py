import numpy as np
import os
import datetime
import game_generator
from psro_trainer import PSRO_trainer
from utils import *
from meta_strategies import double_oracle,fictitious_play

meta_game = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]])

meta_games = [meta_game, -meta_game]

empirical_games = [[0,1,1], [0,1,1]]

dev_strs, nashconv = double_oracle(meta_games, empirical_games, './')
print(dev_strs)
print(nashconv)


dev_strs, nashconv = fictitious_play(meta_games, empirical_games, './')
print(dev_strs)
print(nashconv)
