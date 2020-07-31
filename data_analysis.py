import os
from nash_solver.gambit_tools import load_pkl
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

load_path = os.getcwd() + '/data/data1/'
zero_sum_DO = load_pkl(load_path + 'zero_sum_DO.pkl')
zero_sum_FP = load_pkl(load_path + 'zero_sum_FP.pkl')
zero_sum_DO_FP = load_pkl(load_path + 'zero_sum_DO_SP.pkl')

zero_sum_DO = np.mean(zero_sum_DO, axis=0)
zero_sum_FP = np.mean(zero_sum_FP, axis=0)
zero_sum_DO_FP = np.mean(zero_sum_DO_FP, axis=0)

# idx = 6
# zero_sum_DO = zero_sum_DO[idx]
# zero_sum_FP = zero_sum_FP[idx]
# zero_sum_DO_FP = zero_sum_DO_FP[idx]

# Focus on fictitious play
# fic_zero_sum_DO_FP = []
# for i in range(len(zero_sum_DO_FP)):
#     if i % 2 == 1:
#         fic_zero_sum_DO_FP.append(zero_sum_DO_FP[i])
# y = np.arange(1, len(zero_sum_DO)+1, 2)
# plt.plot(y, fic_zero_sum_DO_FP, '-C1', label= "DO+FP")

x = np.arange(1, len(zero_sum_DO)+1)
plt.plot(x, zero_sum_DO, '-C2', label= "DO")
plt.plot(x, zero_sum_FP, '-C0', label= "FP")
plt.plot(x, zero_sum_DO_FP, '-C1', label= "DO+FP")



plt.xlabel("Number of Iterations")
plt.ylabel("NashConv")
plt.title("Average NashConv over 30 runs in Synthetic Zero-Sum Game")
plt.legend(loc="best")
plt.show()

plt.show()
