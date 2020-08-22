import numpy as np

def smoothing_kl(p, q, eps=0.001):
    p = smoothing(p, eps)
    q = smoothing(q, eps)
    return np.sum(p * np.log(p / q))


def smoothing(p, eps):
    p = np.array(p, dtype=np.float)
    zeros_pos_p = np.where(p == 0)[0]
    num_zeros = len(zeros_pos_p)
    x = eps * num_zeros / (len(p) - num_zeros)
    for i in range(len(p)):
        if i in zeros_pos_p:
            p[i] = eps
        else:
            p[i] -= x
    return p

class Exp3(object):
    """
    EXP3 algorithm for adversarial bandit.
    """
    def __init__(self,
                 num_arms,
                 num_players,
                 gamma=0.0,
                 kl_coef=1,
                 abs_value=False,
                 kl_regularization=False
                 ):
        self.weights = np.ones(num_arms)
        self.num_arms = num_arms
        self.num_players = num_players
        self.gamma = gamma
        self.arm_pulled = 0
        self.abs_value = abs_value
        self.kl_regularization = kl_regularization
        self.kl_coef = kl_coef

    def sample(self, temerature=None):
        """
        Sample a new arm to pull.
        :return: int, index of arms.
        """
        weight_sum = np.sum(self.weights)
        self.probability_distribution = [(1.0 - self.gamma) * (w / weight_sum) + (self.gamma / len(self.weights)) for w in self.weights]
        self.arm_pulled = np.random.choice(range(len(self.probability_distribution)), p=self.probability_distribution)
        return self.arm_pulled

    def update_weights(self, reward):
        rewards = np.zeros(self.num_arms)
        rewards[self.arm_pulled] = reward/self.probability_distribution[self.arm_pulled]
        self.weights *= np.exp(rewards * self.gamma / self.num_arms)

def softmax(x, temperature=1/1.3):
    return np.exp(x / temperature)/np.sum(np.exp(x / temperature))

class pure_exp(object):
    def __init__(self,
                 num_arms,
                 num_players,
                 gamma=0.0,
                 slow_period=None,
                 fast_period=None,
                 kl_coef=0.1,
                 abs_value=False,
                 kl_regularization=False):
        self.weights = np.ones(num_arms) * 100
        self.num_arms = num_arms
        self.num_players = num_players
        self.gamma = gamma
        self.arm_pulled = 0
        self.abs_value = abs_value
        self.kl_regularization = kl_regularization
        self.kl_coef = kl_coef
        self.slow_period = slow_period
        self.fast_period = fast_period

    def sample(self, num_iters):
        temperature = self.temperature_scheme(num_iters)
        self.probability_distribution = softmax(self.weights, temperature=temperature)
        self.arm_pulled = np.random.choice(range(len(self.probability_distribution)), p=self.probability_distribution)
        return self.arm_pulled

    def update_weights(self, reward, NE_list=None):
        if self.abs_value:
            reward = abs(reward)
        # if self.kl_regularization:
        #     kl_conv = self.calculate_kl(NE_list)
        #     reward += self.kl_coef * kl_conv
        self.weights[self.arm_pulled] = (1 - self.gamma) * reward + self.gamma * self.weights[self.arm_pulled]

    def temperature_scheme(self, num_iters):
        # Numbers are hyperparameters.
        if num_iters < 20:
            return 1
        elif num_iters < 30:
            return 5
        else:
            return 10

    def calculate_kl(self, NE_list):
        if len(NE_list) <= 2 * (self.slow_period + self.fast_period):
            return 0
        kl_conv = 0
        for player in range(self.num_players):
            p = NE_list[-(2 + self.slow_period + self.fast_period)][player]
            q = NE_list[-2][player]
            p = np.append(p, [0] * (len(q)-len(p)))
            kl = smoothing_kl(p, q)
            kl_conv += kl
        return kl_conv
