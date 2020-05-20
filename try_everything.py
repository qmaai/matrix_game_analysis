import numpy as np
import os
import datetime
import game_generator
from psro_trainer import PSRO_trainer
from meta_strategies import mixed_strategy_payoff, deviation_strategy, double_oracle, fictitious_play

meta_game = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]])

meta_games = [meta_game, -meta_game]
empirical_game = [[0,1], [0,1]]
dev_strs, nashconv = fictitious_play(meta_games, empirical_game, './')
print(dev_strs)
print(nashconv)
# probs = [np.array([1, 0, 0]), np.array([0, 0, 1])]
# dev_strs, dev_payoff = deviation_strategy(meta_games, probs)
# print(dev_strs)
# print(dev_payoff)


