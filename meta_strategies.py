import numpy as np
import collections
from nash_solver.general_nash_solver import gambit_solve
from minimum_regret_profile import minimum_regret_profile_calculator
from utils import *

def double_oracle(meta_games, empirical_games, checkpoint_dir):
    num_players = len(meta_games)
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])
    nash = gambit_solve(subgames, mode="one", checkpoint_dir=checkpoint_dir[:-1])
    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)
    nashconv = 0
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv



def fictitious_play(meta_games, empirical_games, checkpoint_dir=None):
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []
    counter0 = collections.Counter(empirical_games[0])
    counter1 = collections.Counter(empirical_games[1])

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    nash0 = np.ones(len(idx0))
    for i, item in enumerate(idx0):
        nash0[i] = counter0[item]
    nash0 /= np.sum(nash0)

    nash1 = np.ones(len(idx1))
    for i, item in enumerate(idx1):
        nash1[i] = counter1[item]
    nash1 /= np.sum(nash1)
    nash = [nash0, nash1]
    
    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)

    nashconv = 0
    for player in range(len(meta_games)):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv

def mrcp_solver(meta_games, empirical_games, checkpoint_dir=None, recursive=False):
    """
    A wrapper for minimum_regret_profile_calculator, automatically test iterations and clearning remnants mrcp values
    """
    if not hasattr(mrcp_solver, "mrcp_calculator"):
        mrcp_solver.mrcp_calculator  = minimum_regret_profile_calculator(full_game=meta_games, recursive=recursive)
    else:
        # test full game the same
        full_game_different = meta_games[0].shape != mrcp_solver.mrcp_calculator.full_game[0].shape or np.sum(np.absolute(meta_games[0]-mrcp_solver.mrcp_calculator.full_game[0]),axis=None) != 0
        if full_game_different: # change mrcp_calculator
            print('changing mrcp calculator!!!')
            mrcp_solver.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games, recursive=recursive)
        elif mrcp_solver.mrcp_calculator.mrcp_empirical_game is not None and len(empirical_games[0])<=len(mrcp_solver.mrcp_calculator.mrcp_empirical_game[0]):
            # another round of random start from full game, might exsit potential bugs
            # as I changed _last_empirical_game to mrcp_empirical_game
            print('clearing the past data from mrcp calculator, should be at the start of a new round')
            mrcp_solver.mrcp_calculator.clear()
        else:
            pass

    mrcp_solver.mrcp_calculator(empirical_games)

    num_strategies = meta_games[0].shape[0]
    idx0 = sorted(list(set(mrcp_solver.mrcp_calculator.mrcp_empirical_game[0])))
    idx1 = sorted(list(set(mrcp_solver.mrcp_calculator.mrcp_empirical_game[1])))

    meta_game_nash = []
    for i,idx in enumerate([idx0,idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, mrcp_solver.mrcp_calculator.mrcp_profile[i])
        meta_game_nash.append(ne)
    
    # find deviation that is not in the empirical game
    dev_strs = []
    prob1 = meta_game_nash[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = meta_game_nash[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    # mask elements inside empirical game
    payoff_vec[list(set(empirical_games[0]))] = -1e5
    dev_strs.append(np.argmax(payoff_vec))

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    payoff_vec[list(set(empirical_games[1]))] = -1e5
    dev_strs.append(np.argmax(payoff_vec))

    return dev_strs, mrcp_solver.mrcp_calculator.mrcp_value
