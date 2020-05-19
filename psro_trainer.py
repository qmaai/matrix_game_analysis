import numpy as np
from exploration import pure_exp

class PSRO_trainer(object):
    def __init__(self, meta_games,
                 num_strategies,
                 num_rounds,
                 meta_method,
                 checkpoint_dir,
                 meta_method_list=None,
                 blocks=False):
        self.meta_games = meta_games
        self.num_rounds = num_rounds
        self.meta_method = meta_method
        self.num_strategies = num_strategies
        self.checkpoint_dir = checkpoint_dir
        self.meta_method_list = meta_method_list
        self.mode = 1
        self.blocks = blocks

        self.empirical_games = [[], []]
        self.num_iterations = 0
        self.num_used_iterations = []

        self.fast_period = 3
        self.slow_period = 1
        self.fast_count = 3
        self.slow_count = 1

        self.selector = pure_exp(2,
                                 2,
                                slow_period=self.slow_period,
                                fast_period=self.fast_period,
                                abs_value=True)
        self.selector.arm_pulled = 0

    def init_round(self):
        init_strategy = np.random.randint(0, self.num_strategies)
        self.empirical_games = [[init_strategy], [init_strategy]]
        self.num_iterations = 0
        self.mode = 0

        if self.blocks:
            nash_payoff = self.meta_games[init_strategy, init_strategy]
            nashconv = np.max(self.meta_games[:, init_strategy]) + np.max(self.meta_games[init_strategy, :]) - nash_payoff
            self.blocks_nashconv = [nashconv]

    def iteration(self):
        while True:
            self.num_iterations += 1
            dev_strs, _ = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            if dev_strs[0] in self.empirical_games[0] and dev_strs[1] in self.empirical_games[1]:
                self.num_used_iterations.append(self.num_iterations)
            if dev_strs[0] not in self.empirical_games[0] and dev_strs[0] is not None:
                self.empirical_games[0].append(dev_strs[0])
            if dev_strs[1] not in self.empirical_games[1] and dev_strs[1] is not None:
                self.empirical_games[1].append(dev_strs[1])
            if self.meta_method_list is not None:
                self.mode = 1 - self.mode
                self.meta_method = self.meta_method_list[self.mode]

    def loop(self):
        for _ in range(self.num_rounds):
            self.init_round()
            if self.blocks:
                self.iteration_blocks()
            else:
                self.iteration()

    # For blocks
    def iteration_blocks(self):
        while True:
            self.num_iterations += 1
            dev_strs, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            if not self.mode:
                self.blocks_nashconv.append(nashconv)
            if dev_strs[0] in self.empirical_games[0] and dev_strs[1] in self.empirical_games[1]:
                self.num_used_iterations.append(self.num_iterations)
            if dev_strs[0] not in self.empirical_games[0]:
                self.empirical_games[0].append(dev_strs[0])
            if dev_strs[1] not in self.empirical_games[1]:
                self.empirical_games[1].append(dev_strs[1])
            if self.meta_method_list is not None:
                if self.mode:
                    self.fast_count -= 1
                    if self.fast_count == 0:
                        self.fast_count = self.fast_period
                        self.mode = 1 - self.mode
                        self.meta_method = self.meta_method_list[0]
                else:
                    self.slow_count -= 1
                    if self.slow_count == 0:
                        self.slow_count = self.fast_period
                        self.mode = 1 - self.mode
                        self.selector.update_weights(self.blocks_nashconv[-1]-self.blocks_nashconv[-2])
                        next_method = self.selector.sample(self.num_iterations)
                        self.meta_method = self.meta_method_list[next_method]
