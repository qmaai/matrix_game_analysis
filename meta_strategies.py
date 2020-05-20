import numpy as np
import collections
from nash_solver.general_nash_solver import gambit_solve

def mixed_strategy_payoff(meta_games, probs):
    payoffs = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]
    for meta_game in meta_games:
        payoffs.append(np.sum(prob1 * meta_game * prob2))
    return payoffs

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
    num_players = len(meta_games)
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
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv
