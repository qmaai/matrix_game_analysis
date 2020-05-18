import numpy as np

class Psro_trainer(object):
    def __init__(self, meta_games,
                 num_strategies,
                 num_rounds,
                 meta_method,
                 checkpoint_dir):
        self.meta_games = meta_games
        self.num_rounds = num_rounds
        self.meta_method = meta_method
        self.num_strategies = num_strategies
        self.checkpoint_dir = checkpoint_dir

        self.empirical_games = [[], []]
        self.num_iterations = 0
        self.num_used_iterations = []

    def init_round(self):
        init_strategy = np.random.randint(0, self.num_strategies)
        self.empirical_games = [[init_strategy], [init_strategy]]
        self.num_iterations = 0

    def iteration(self):
        while True:
            self.num_iterations += 1
            dev_strs = self.meta_games(self.meta_games, self.empirical_games, self.checkpoint_dir)
            if dev_strs[0] in self.empirical_games[0] and dev_strs[1] in self.empirical_games[1]:
                self.num_used_iterations.append(self.num_iterations)
            if dev_strs[0] not in self.empirical_games[0]:
                self.empirical_games[0].append(dev_strs[0])
            if dev_strs[1] not in self.empirical_games[1]:
                self.empirical_games[1].append(dev_strs[1])

    def loop(self):
        for _ in range(self.num_rounds):
            self.init_round()
            self.iteration()
