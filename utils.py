import numpy as np
import random
from open_spiel.python.algorithms.psro_v2 import meta_strategies

def set_random_seed(seed=None):
    seed = np.random.randint(low=0,high=1e5) if seed is None else seed
    np.random.seed(seed)
    random.seed(seed)
    return seed


def mixed_strategy_payoff(meta_games, probs):
    """
    A multiple player version of mixed strategy payoff writen below by yongzhao
    The lenth of probs could be smaller than that of meta_games
    """
    assert len(meta_games)==len(probs),'number of player not equal'
    for i in range(len(meta_games)):
        assert len(probs[i]) <= meta_games[0].shape[i],'meta game should have larger dimension than marginal probability vector'
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(probs)
    prob_slice = tuple([slice(prob_matrix.shape[i]) for i in range(len(meta_games))])
    meta_game_copy = [ele[prob_slice] for ele in meta_games]
    payoffs = []
    for i in range(len(meta_games)):
        payoffs.append(np.sum(meta_game_copy[i]*prob_matrix))
    return payoffs

# This older version of function must be of two players
#def mixed_strategy_payoff(meta_games, probs):
#    payoffs = []
#    prob1 = probs[0]
#    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
#    prob2 = probs[1]
#    for meta_game in meta_games:
#        payoffs.append(np.sum(prob1 * meta_game * prob2))
#    return payoffs

def regret_of_variable(prob_var, empirical_games, meta_game):
    """
    Only works for two player case
    Calculate the function value of one data point prob_var
    in amoeba method, Reshape and expand the probability var into full shape
    Input:
        prob_var       : variable that amoeba directly search over
        empirical_games: a list of list, indicating player's strategy sets
        meta_game      : the full game matrix to calculate deviation from
    """
    probs = []
    index = np.cumsum([len(ele) for ele in empirical_games])
    pointer = 0
    for i, idx in enumerate(empirical_games):
        prob = np.zeros(meta_game[0].shape[i])
        np.put(prob, idx, prob_var[pointer:index[i]])
        pointer = index[i]
        probs.append(prob)

    _, dev_payoff = deviation_strategy(meta_game,probs) 
    payoff = mixed_strategy_payoff(meta_game, probs)
    return sum(dev_payoff)-sum(payoff)


def deviation_strategy(meta_games, probs):
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    return dev_strs, dev_payoff


