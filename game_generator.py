import numpy as np

class Game_generator(object):
    def __init__(self,
                 num_strategies,
                 num_players=2,
                 payoff_ub=11,
                 payoff_lb=-10):
        self.num_strategies = num_strategies
        self.num_players = num_players
        assert num_players == 2
        self.payoff_ub = payoff_ub
        self.payoff_lb = payoff_lb

    def zero_sum_game(self):
        meta_game = np.random.randint(low=self.payoff_lb,
                                      high=self.payoff_ub,
                                      size=(self.num_strategies, self.num_strategies))
        return [meta_game, -meta_game]

    def general_sum_game(self):
        meta_game1 = np.random.randint(low=self.payoff_lb,
                                      high=self.payoff_ub,
                                      size=(self.num_strategies, self.num_strategies))
        meta_game2 = np.random.randint(low=self.payoff_lb,
                                       high=self.payoff_ub,
                                       size=(self.num_strategies, self.num_strategies))
        return [meta_game1, meta_game2]

    def symmetric_zero_sum_game(self):
        meta_game = np.random.randint(low=self.payoff_lb/2,
                                      high=np.ceil(self.payoff_ub/2),
                                      size=(self.num_strategies, self.num_strategies))
        meta_game += meta_game.T
        return [meta_game, -meta_game]

    def transitive_game(self):
        raise NotImplementedError

    def cyclic_game(self):
        raise NotImplementedError
