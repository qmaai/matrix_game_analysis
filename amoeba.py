from math import sqrt
import numpy as np
from functools import partial
from utils import regret_of_variable 

# Amoeba uses the simplex method of Nelder and Mead to maximize a
# function of 1 or more variables, constraints are put into place
# according to Patrick Jordan's Thesis "practical strategic reaso
# ning with applications in market games" section 7.2
#   Copyright (C) 2020  Gary Qiurui Ma, Strategic Reasing Group

def check_within_probability_simplex(var):
    '''
    check variable is within nprobability simplex
    '''
    return np.all(var>=0) and np.all(var<=1)

def shrink_simplex(sorted_simplex, sorted_value, func, rho = 0.5):
    """
    Please beware that simplex and function value supplied must be sorted
    based on the function value, the smallest value at the front
    Input:
        sorted_simplex: the simplex of length nvar+1. The first the best
        sorted_value  : the fvalue of simplexes
        func          : function to evaluate the points
        rho           : hyperparamter to shink the simplex
    """
    assert len(sorted_simplex) == len(sorted_value)
    simplex, fvalue = [sorted_simplex[0]], [sorted_value[0]]
    for i in range(1,len(sorted_simplex)):
        simplex.append(simplex[0]+rho*(sorted_simplex[i]-simplex[0])) 
        fvalue.append(func(simplex[i]))
    return simplex, fvalue 

def amoeba_mrcp(empirical_game, full_game, var='uni', max_iter=5000, ftolerance=1.e-4, xtolerance=1.e-4):
    """
    Note each varibale in the amoeba variable is two times the length of the strategies
    Input:
        empirical_game : each player's strategy set
        full_game      : the full meta game to compute mrcp on
        var            : initial guessing for the solution. defaulted to uniform
        max_iter       : maximum iteration of amoeba to automatically end
        ftolerance     : smallest difference of best and worst vertex to converge
        xtolerance     : smallest difference in average point and worst point of simplex
    """
    def normalize(sections, variables):
        """
        A variable made of len(sections) parts, each of the parts is
        in a probability simplex
        Input:
            variables: the varible that amoeba is searching through
            sections : a list containing number of element for each section.
                       Typically it is the list of number of strategies
        Output:
            A normalized version of the varibales by sections
        """
        pointer = 0
        for ele in np.cumsum(sections):
            variables[pointer:ele] /= sum(variables[pointer:ele])
            pointer = ele
        return variables

    # construct function for query
    func = partial(regret_of_variable,
            empirical_games=empirical_game,
            meta_game=full_game) 

    sections = [len(ele) for ele in empirical_game]    # num strategies for players
    normalize = partial(normalize, sections=sections)  # force into simplex
    if var=='uni':
        var = np.ones(sum(sections))      # the initial point of search from uniform
    elif var=='rand': # random initial points
        var = np.random.rand(sum(sections))
    else:
        assert len(var) == sum(sections), 'initial points incorrect shape'

    var = normalize(variables=var)       

    nvar = sum(sections)                  # total number of variables to minimize over
    nsimplex = nvar + 1                   # number of points in the simplex

    # set up the simplex, the first point is the guess. all sides of simplex
    # have length |c|. Please tweak this value should constraints be violated
    # assume if vertexes on simplex is normalized, then reflection, expansion
    # shrink will be on the probability simplex
    c = 1
    val_b = c/nvar/sqrt(2)*(sqrt(nvar+1)-1)
    val_a = val_b + c/sqrt(2)

    simplex = [0]*nsimplex
    simplex[0] = var[:]
    
    for i in range(nvar):
        addition_vector = np.ones(sum(sections))*val_b
        addition_vector[i] = val_a
        simplex[i+1] = normalize(variables=simplex[0]+addition_vector)

    fvalue = []
    for i in range(nsimplex):  # set the function values for the simplex
        fvalue.append(func(simplex[i]))
    
    iteration = 0
    #print("##############################")
    #print("##############################")
    while iteration < max_iter:

        # sort the simplex and the fvalue the last one is the worst
        sort_index = np.argsort(fvalue)
        fvalue = [fvalue[ele] for ele in sort_index]
        simplex = [simplex[ele] for ele in sort_index]
        #print()
        #print("##############################")
        #print("start of iteration {} with empirical game".format(iteration),empirical_game)
        #print('simplex : ',simplex)
        #print('fvalue  : ',fvalue)
            
        # get the average of the the n points except from the worst
        x_a = np.average(np.array(simplex[:-1]),axis=0)
        assert check_within_probability_simplex(x_a), 'centroid not in probability simplex'

        # determine the termination criteria
        # 1. distance between average and worst
        simscale = np.sum(np.absolute(x_a-simplex[-1]))/nvar
        # 2. distance between best and worst function values
        fscale = (abs(fvalue[0])+abs(fvalue[-1]))/2.0
        if fscale != 0.0:
            frange = abs(fvalue[0]-fvalue[-1])/fscale
        else:
            frange = 0.0  # all the fvalues are zero in this case
        
        #print('frange',frange)
        #print('simscale',simscale)
        # have we converged?
        if (ftolerance <= 0.0 or frange < ftolerance) \
                and (xtolerance <= 0.0 or simscale < xtolerance):
            #print('amoeba finished with {} iteration'.format(iteration))
            return np.split(simplex[0],sections[:-1]),fvalue[0],iteration

        # perform reflection to acquire x_r,evaluate f_r
        alpha = 1
        x_r = x_a + alpha*(x_a-simplex[-1])
        while not check_within_probability_simplex(x_r):
            alpha /= 2
            x_r = x_a + alpha*(x_a-simplex[-1])
        f_r = func(x_r)
        #print('centroid',x_a)
        #print('reflection point',x_r,f_r)
        # expansion if the reflection is better
        if f_r < fvalue[0]:    # expansion if the reflection is better
            gamma = 1
            x_e = x_r + gamma*(x_r-x_a)
            while not check_within_probability_simplex(x_e):
                gamma /= 2
                x_e = x_r + gamma*(x_r-x_a)
            f_e = func(x_e)
            if f_e < fvalue[0]: # accept expansion and replace the worst point
                simplex[-1] = x_e
                fvalue[-1] = f_e
                #print('accept expansion, better than best')
            else:               # refuse expansion and accept reflection
                simplex[-1] = x_r
                fvalue[-1] = f_r
                #print('reject expansion, accept reflection, worse than best')
        elif f_r < fvalue[-2]:  # accept reflection when better than lousy
            simplex[-1] = x_r
            fvalue[-1] = f_r
            #print('accept reflection, better than lousy')
        else:
            if f_r > fvalue[-1]: # inside contract if reflection is worst than worst
                x_c = x_a - 0.5*(x_a-simplex[-1]) # 0.5 being a hyperparameter
                f_c = func(x_c)
                if f_c < fvalue[-1]: # accept inside contraction
                    simplex[-1] = x_c
                    fvalue[-1] = f_c
                    #print('accept inside contract, better than worse')
                else:
                    simplex, fvalue = shrink_simplex(simplex,fvalue,func)
                    #print('reject inside contract, shrinked the simplex')
            else:                # outside contract if reflection better than worse
                x_c = x_a + alpha*0.5*(x_a-simplex[-1]) # 0.5 being a hyperparameter
                f_c = func(x_c)
                if f_c < f_r:    # accept contraction
                    simplex[-1] = x_c
                    fvalue[-1] = f_c
                    #print('accept outside contract, better than reflection')
                else:
                    simplex, fvalue = shrink_simplex(simplex,fvalue,func)
                    #print('reject outside contract, shrinked the simplex')
        #print('iteration {}:'.format(iteration),end=' ')
        #print('new fvalue',fvalue)
        iteration += 1            
    #print('amoeba out of {} iteration'.format(iteration))
    sort_index = np.argsort(fvalue)
    fvalue = [fvalue[ele] for ele in sort_index]
    simplex = [simplex[ele] for ele in sort_index]
    return np.split(simplex[0],sections[:-1]), fvalue[0], iteration
