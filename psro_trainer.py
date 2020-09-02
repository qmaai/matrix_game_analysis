import numpy as np
from utils import set_random_seed
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
                 blocks=False,
                 seed=None,
                 calculate_neconv=True,
                 calculate_mrcpconv=True,
                 init_strategies=None):
        """
        Inputs:
            num_rounds      : repeat psro on matrix games from #num_rounds start points
            meta_method_list: for heuristics block switching
            blocks          : HBS
            seed            : a integer. If provided, reset every round to guarantee that mrcp calculation is deterministic given an empirical game:
            calculate_neconv   : ne_conv to evaluate to evaluate the heuristics
            calculate_mrcpconv : mrcp_conv to evaluate the heuristics
            init_strategies    : a len(num_rounds) list or a number
        """
        self.meta_games = meta_games
        self.num_rounds = num_rounds
        self.meta_method = meta_method
        self.num_strategies = num_strategies
        self.checkpoint_dir = checkpoint_dir
        self.meta_method_list = meta_method_list
        self.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games)
        self.mode = 0
        self.blocks = blocks
        self.seed = seed
        self.calculate_neconv= calculate_neconv
        self.calculate_mrconv= calculate_mrcpconv
        if init_strategies is not None:
            assert isinstance(init_strategies,int) or len(init_strategies)==num_rounds, \
                    "provide initial strategies with right length"
            self.init_strategies = np.array([init_strategies for _ in range(num_rounds)],dtype=int) if isinstance(init_strategies,int) else np.array(init_strategies,dtype=int)
        else:
            self.init_strategies = np.random.randint(0, num_strategies, num_rounds)

        self.empirical_games = [[], []]
        self.num_iterations = num_iterations

        self.fast_period = 1
        self.slow_period = 1
        self.fast_count = 1
        self.slow_count = 1
        self.nashconvs = []  # Heuristic-conv. The name is retained for convenience sake
        self.neconvs = []
        self.mrconvs = []
        self.mrprofiles = []

    def init_round(self,init_strategy):
        #init_strategy = np.random.randint(0, self.num_strategies)
        #init_strategy = 0
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
        neconv_list = []
        mrconv_list = []
        mrprofile_list = []

        # Tricky Detail: mrcp does not calculate NE's first empirical game's mrcp value
        # ne does not calculate mrcp's first empirical game's NE-based regret
        if self.calculate_mrconv:
            if self.meta_method.__name__!='mrcp_solver':
                mrcp_profile, mrcp_value = self.mrcp_calculator(self.empirical_games)
                mrconv_list.append(mrcp_value)
                mrprofile_list.append(mrcp_profile)
        if self.meta_method.__name__!='double_oracle':
            _, neconv = double_oracle(self.meta_games,self.empirical_games,self.checkpoint_dir)
            neconv_list.append(neconv)

        for it in range(self.num_iterations):
            print('##################Iteration {}###############'.format(it))
            dev_strs, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            nashconv_list.append(nashconv)
            self.empirical_games[0].append(dev_strs[0])
            self.empirical_games[0] = sorted(self.empirical_games[0])
            self.empirical_games[1].append(dev_strs[1])
            self.empirical_games[1] = sorted(self.empirical_games[1])
            if self.meta_method_list is not None:
                self.mode = 1 - self.mode
                self.meta_method = self.meta_method_list[self.mode]

            if self.calculate_neconv:
                if self.meta_method.__name__!='double_oracle':
                    _, neconv = double_oracle(self.meta_games,
                                              self.empirical_games,
                                              self.checkpoint_dir)
                    neconv_list.append(neconv)
                else:
                    neconv_list.append(nashconv) 

            if self.calculate_mrconv:
                if self.meta_method.__name__!='mrcp_solver':
                    mrcp_profile, mrcp_value = self.mrcp_calculator(self.empirical_games)
                    mrconv_list.append(mrcp_value)
                    mrprofile_list.append(mrcp_profile)
                else:
                    mrconv_list.append(nashconv)
                    mrprofile_list.append(self.meta_method.mrcp_calculator.mrcp_profile)
        
        # Tricky part: Nashconv does not add the last value after update
        # mrcp does not add its last own value after its last update
        # NE does not add its last own value after its last update
        _,nashconv = self.meta_method(self.meta_games,self.empirical_games,self.checkpoint_dir)
        nashconv_list.append(nashconv)
        if self.meta_method.__name__=='mrcp_solver':
            mrconv_list.append(nashconv)
            mrprofile_list.append(self.meta_method.mrcp_calculator.mrcp_profile)
        if self.meta_method.__name__=='double_oracle':
            neconv_list.append(nashconv)
        
        self.nashconvs.append(nashconv_list)
        self.mrconvs.append(mrconv_list)
        self.mrprofiles.append(mrprofile_list)
        self.neconvs.append(neconv_list)

    def loop(self):
        for i in range(self.num_rounds):
            self.init_round(self.init_strategies[i])
            # reset to same random seed to guarantee MRCP
            # being deterministic given empirical game
            if self.seed is not None:
                set_random_seed(self.seed) 
            if self.blocks:
                self.iteration_blocks()
            else:
                self.iteration()

            self.mrcp_calculator.clear()

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
