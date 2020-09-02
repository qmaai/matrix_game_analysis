import numpy as np
from functools import partial
from itertools import chain, combinations,product

from amoeba import amoeba_mrcp

class minimum_regret_profile_calculator(object):
    """
    Implement MRCP in Pjordan's thesis Algorithm 8 FIND-MRCP
    Assume the mimimum_regret_profile_calculator is called every iteration of PSRO
    applicable to multiple player case.
    """
    def __init__(self, full_game, recursive=False):
        """
        Input:
            full_game     : full matrix game to calculate regret
            recursive     : explore all subgames in the restricted index
        """
        self.full_game = full_game
        self.no_derivative_opt_method = partial(amoeba_mrcp, full_game=full_game)
        self.recursive = recursive
        # mrcp_profile and mrcp_value records the last iteration's meta game's
        # minimum regret profile and value, which corresponds to last_empirical_game.
        # Anything besides it in the past history does not need to be
        # recorded. As the newest iteration could only bring about changes where
        # the new strategies are concerned
        self._last_empirical_game = None
        self._mrcp_iteration = 0
        self.mrcp_profile = []
        self.mrcp_value = 1e5 #beware of the game who's payoff is even larger
        self.mrcp_empirical_game = None # documents the empircal game for the last mrcp

    def clear(self):
        self._last_empirical_game = None
        self._mrcp_iteration = 0
        self.mrcp_profile = []
        self.mrcp_value = 1e5
        self.mrcp_empirical_game = None

    def __call__(self, empirical_game):
        # for comparing pjordan recursive and non-recursive ones
        #profile_recursive, value_recursive = self.recursive_find_mrcp(empirical_game)
        #self.clear()
        #profile, value = self.find_mrcp(empirical_game)
        #print('recursive {}  VS normal {}'.format(value_recursive,value))
        #print('profile_recursive', profile_recursive)
        #print('profile',profile)
        #return profile_recursive, value_recursive

        if self.recursive:
            return self.recursive_find_mrcp(empirical_game)
        else:
            return self.find_mrcp(empirical_game)
    
    def find_mrcp(self, empirical_game, repeat=100):
        '''
        Implement ways in 'analyzing complex strategc interactions in multiagent systems' by Walsh el.
        Run amoeba number of times until the same mrcp has been found for more than 10 times
        Input:
            empirical_game: the strategy set players have at this iteration of p
            repeat        : number of different initial points(different trials) to start with
        '''
        # first remove duplicate from empirical game
        empirical_game = [sorted(list(set(ele))) for ele in empirical_game]
        
        for iters in range(repeat):
            # initiate_starting_point
            mrcp_mixed_strategy, mrcp_value, iteration = self.no_derivative_opt_method(empirical_game, var='rand')
            if self.mrcp_value > mrcp_value:
                self.mrcp_value = mrcp_value
                self._mrcp_iteration = iteration
                self.mrcp_empirical_game = empirical_game
                self.mrcp_profile = mrcp_mixed_strategy
        print('iteration {} mrcp value {} profile {}'.format(self._mrcp_iteration,self.mrcp_value,self.mrcp_profile))
        return self.mrcp_profile, self.mrcp_value

    def recursive_find_mrcp(self, empirical_game):
        '''
        Get all possible subgame indexes for strategies in meta game, implements Algorithm 8 in jordan's paper
        Input:
            empirical_game: the strategy set players have at this iteration of psro
        '''
        # first remove duplicate from empirical game
        empirical_game = [sorted(list(set(ele))) for ele in empirical_game]

        strategy_indexes = [] # strategy combination for each player
        if self._last_empirical_game == None: # find all subgame indexes
            self._num_players = len(empirical_game)
            for p in range(self._num_players):
                strategy_indexes.append(list(chain(*map(lambda x: \
                        combinations(empirical_game[p], x), range(1,len(empirical_game[p])+1)))))
            # all possible subgame indexes
            indexes = list(product(*strategy_indexes))
        else: # find all subgame indexes where new strategies exist
            # test if no new strategies is added
            no_new_strategies = True
            for p in range(self._num_players):
                old_strategies = self._last_empirical_game[p]
                new_strategies = [s for s in empirical_game[p] \
                        if s not in old_strategies]
                old_combo = list(chain(*map(lambda x:combinations(old_strategies, x),\
                        range(0,len(old_strategies)+1)))) # auxillary old strategies
                new_combo = list(chain(*map(lambda x:combinations(new_strategies, x),\
                        range(1,len(new_strategies)+1)))) #make sure new strategies exist
                if len(new_combo)==0: # one player no new strategies but others do
                    # old_combo could include emty strategy to cater for the
                    # supposedly un-empty new strategy list
                    old_combo = [ele for ele in old_combo if len(ele)>0]
                    strategy_indexes.append(old_combo)
                else:
                    no_new_strategies = False
                    strategy_indexes.append([ele[0]+ele[1] \
                            for ele in list(product(old_combo,new_combo))])
            if no_new_strategies:
                indexes = []
            else:
                indexes = list(product(*strategy_indexes))

        # traverse the index of all new element subgames and perfome amoeba
        print()
        print("########################################")
        print("strategy set ",empirical_game, end = " ")
        print("has {} of new subgames".format(len(indexes)))
        for ind in indexes:
            mrcp_mixed_strategy, mrcp_value, iteration = self.no_derivative_opt_method(ind)
            if self.mrcp_value > mrcp_value:
                self.mrcp_value = mrcp_value
                self._mrcp_iteration = iteration
                self.mrcp_empirical_game = empirical_game
                # reconstruct the mrcp mixed strategy position
                # return on the restricted game view the probability
                # support vector of the mrcp strategy
                self.mrcp_profile.clear()
                li_ind = list([list(ele) for ele in ind])
                for p in range(self._num_players):
                    self.mrcp_profile.append([mrcp_mixed_strategy[p][li_ind[p].index(ele)] if ele in li_ind[p] else 0 for ele in empirical_game[p]])

        self._last_empirical_game = empirical_game
        print('iteration',self._mrcp_iteration,'mrcp profile',self.mrcp_profile)
        return self.mrcp_profile, self.mrcp_value
