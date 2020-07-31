import numpy as np
from exploration import pure_exp
from meta_strategies import double_oracle
from minimum_regret_profile import minimum_regret_profile_calculator

class PSRO_trainer(object):
    def __init__(self, meta_games,
                 num_strategies,
                 num_rounds,
                 meta_method,
                 checkpoint_dir,
                 meta_method_list=None,
                 num_iterations=20,
                 blocks=False):
        self.meta_games = meta_games
        self.num_rounds = num_rounds
        self.meta_method = meta_method
        self.num_strategies = num_strategies
        self.checkpoint_dir = checkpoint_dir
        self.meta_method_list = meta_method_list
        self.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games)
        self.mode = 0
        self.blocks = blocks

        self.empirical_games = [[], []]
        self.num_iterations = num_iterations

        self.fast_period = 1
        self.slow_period = 1
        self.fast_count = 1
        self.slow_count = 1
        self.nashconvs = []
        self.mrconvs = []

    def init_round(self):
        init_strategy = np.random.randint(0, self.num_strategies)
        self.empirical_games = [[init_strategy], [init_strategy]]
        self.mode = 0

        if self.blocks:
            nash_payoff = self.meta_games[init_strategy, init_strategy]
            nashconv = np.max(self.meta_games[:, init_strategy]) + np.max(self.meta_games[init_strategy, :]) - nash_payoff
            self.blocks_nashconv = [nashconv]
            self.selector = pure_exp(2,
                                     2,
                                     slow_period=self.slow_period,
                                     fast_period=self.fast_period,
                                     abs_value=True)
            self.selector.arm_pulled = 0

    def iteration(self):
        nashconv_list = []
        mrconv_list = []
        for _ in range(self.num_iterations):
            dev_strs, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            if nashconv is not None:
                nashconv_list.append(nashconv)
            else:
                _, nashconv = double_oracle(self.meta_games, self.empirical_games, self.checkpoint_dir)
                nashconv_list.append(nashconv)
            self.empirical_games[0].append(dev_strs[0])
            self.empirical_games[0] = sorted(self.empirical_games[0])
            self.empirical_games[1].append(dev_strs[1])
            self.empirical_games[1] = sorted(self.empirical_games[1])
            if self.meta_method_list is not None:
                self.mode = 1 - self.mode
                self.meta_method = self.meta_method_list[self.mode]
            _, mrcp_value = self.mrcp_calculator(self.empirical_games)
            mrconv_list.append(mrcp_value)
        self.nashconvs.append(nashconv_list)
        self.mrconvs.append(mrconv_list)

    def loop(self):
        for _ in range(self.num_rounds):
            self.init_round()
            if self.blocks:
                self.iteration_blocks()
            else:
                self.iteration()

    # For blocks
    def iteration_blocks(self):
        nashconv_list = []
        for _ in range(self.num_iterations):
            dev_strs, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            if nashconv is not None:
                nashconv_list.append(nashconv)
                if not self.mode:
                    self.blocks_nashconv.append(nashconv)
            else:
                _, nashconv = double_oracle(self.meta_games, self.empirical_games, self.checkpoint_dir)
                nashconv_list.append(nashconv)
            self.empirical_games[0].append(dev_strs[0])
            self.empirical_games[0] = sorted(self.empirical_games[0])
            self.empirical_games[1].append(dev_strs[1])
            self.empirical_games[1] = sorted(self.empirical_games[1])
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
                        self.slow_count = self.slow_period
                        self.mode = 1 - self.mode
                        self.selector.update_weights(self.blocks_nashconv[-2]-self.blocks_nashconv[-1])
                        next_method = self.selector.sample(self.num_iterations)
                        self.meta_method = self.meta_method_list[next_method]
