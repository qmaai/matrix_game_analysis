from meta_strategies import double_oracle, fictitious_play, mrcp_solver
from game_generator import Game_generator
from psro_trainer import PSRO_trainer
from utils import set_random_seed

from absl import app
from absl import flags
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_rounds", 1, "The number of rounds starting with different.")
flags.DEFINE_integer("num_strategies", 100, "The number of rounds starting with different.")
flags.DEFINE_integer("num_iterations", 40, "The number of rounds starting with different.")
flags.DEFINE_string("game_type", "zero_sum", "Type of synthetic game.")
flags.DEFINE_integer("seed",None,"The seed to control randomness.")
flags.DEFINE_boolean("MRCP_deterministic",True,"mrcp should return a same value given the same empirical game")

def psro(generator,
         game_type,
         num_rounds,
         seed,
         checkpoint_dir,
         meta_method_list=None,
         num_iterations=20,
         blocks=False):
    if game_type == "zero_sum":
        meta_games = generator.zero_sum_game()
    elif game_type == "general_sum":
        meta_games = generator.general_sum_game()
    elif game_type == "symmetric_zero_sum":
        meta_games = generator.general_sum_game()
    else:
        for pkl in os.listdir('efg_game'):
            print(pkl)
            if pkl.split('.pkl')[0] == game_type:
                with open('efg_game/'+pkl,'rb') as f:
                    meta_games = pickle.load(f)
        if not 'meta_games' in locals():
            raise ValueError
    
    # for example 1 in paper
    # meta_games = [np.array([[0,-0.1,-3],[0.1,0,2],[3,-2,0]]),np.array([[0,0.1,3],[-0.1,0,-2],[-3,2,0]])]
    # generator.num_strategies = 3
    # num_rounds = 1
    # num_iterations = 10
    init_strategies = np.random.randint(0,meta_games[0].shape[0],num_rounds)

    DO_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=double_oracle,
                           checkpoint_dir=checkpoint_dir,
                           meta_method_list=meta_method_list,
                           num_iterations=num_iterations,
                           blocks=blocks,
                           seed=seed,
                           init_strategies=init_strategies)

    FP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=fictitious_play,
                           checkpoint_dir=checkpoint_dir,
                           meta_method_list=meta_method_list,
                           num_iterations=num_iterations,
                           blocks=blocks,
                           seed=seed,
                           init_strategies=init_strategies)

    MRCP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=mrcp_solver,
                           checkpoint_dir=checkpoint_dir,
                           meta_method_list=meta_method_list,
                           num_iterations=num_iterations,
                           blocks=blocks,
                           seed=seed,
                           init_strategies=init_strategies)


#    DO_FP_trainer = PSRO_trainer(meta_games=meta_games,
#                              num_strategies=generator.num_strategies,
#                              num_rounds=num_rounds,
#                              meta_method=double_oracle,
#                              checkpoint_dir=checkpoint_dir,
#                              meta_method_list=[double_oracle, fictitious_play],
#                              num_iterations=num_iterations,
#                              blocks=blocks)
#
#    blocks_trainer = PSRO_trainer(meta_games=meta_games,
#                              num_strategies=generator.num_strategies,
#                              num_rounds=num_rounds,
#                              meta_method=double_oracle,
#                              checkpoint_dir=checkpoint_dir,
#                              meta_method_list=[double_oracle, fictitious_play],
#                              num_iterations=num_iterations,
#                              blocks=True)

    DO_trainer.loop()
    print("#####################################")
    print('DO looper finished looping')
    print("#####################################")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + game_type + '_meta_games.pkl','wb') as f:
        pickle.dump(meta_games, f)
    nashconv_names = ['nashconvs_'+str(t) for t in range(len(DO_trainer.neconvs))]
    mrconv_names = ['mrcpcons_'+str(t) for t in range(len(DO_trainer.mrconvs))]
    df = pd.DataFrame(np.transpose(DO_trainer.neconvs+DO_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir+game_type+'_DO.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_DO.pkl','wb') as f:
        pickle.dump(DO_trainer.mrprofiles, f)

    FP_trainer.loop()
    print("#####################################")
    print('FP looper finished looping')
    print("#####################################")
    df = pd.DataFrame(np.transpose(FP_trainer.neconvs+FP_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir+game_type+'_FP.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_FP.pkl','wb') as f:
        pickle.dump(FP_trainer.mrprofiles, f)

    MRCP_trainer.loop()
    print("#####################################")
    print('MRCP looper finished looping')
    print("#####################################")
    df = pd.DataFrame(np.transpose(MRCP_trainer.neconvs+MRCP_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir+game_type+'_MRCP.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_MRCP.pkl','wb') as f:
        pickle.dump(DO_trainer.mrprofiles, f)

#    DO_FP_trainer.loop()
#    blocks_trainer.loop()

    print("The current game type is ", game_type)
    print("DO neco av:", np.mean(DO_trainer.neconvs, axis=0))
    print("DO mrcp av:", np.mean(DO_trainer.mrconvs, axis=0))
    print("FP fpco av:", np.mean(FP_trainer.nashconvs, axis=0))
    print("FP neco av:", np.mean(FP_trainer.neconvs, axis=0))
    print("FP mrcp av:", np.mean(FP_trainer.mrconvs, axis=0))
    print("MR neco av:", np.mean(MRCP_trainer.neconvs, axis=0))
    print("MR mrcp av:", np.mean(MRCP_trainer.mrconvs, axis=0))

#    print("DO+FP average:", np.mean(DO_FP_trainer.nashconvs, axis=0))
#    print("blocks average:", np.mean(blocks_trainer.nashconvs, axis=0))
    print("====================================================")
    

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    seed = set_random_seed(FLAGS.seed)
    if not FLAGS.MRCP_deterministic:
        seed = None # invalidate the seed so it does not get passed into psro_trainer

    generator = Game_generator(FLAGS.num_strategies)
    checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_se_'+str(seed)
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir) + '/'

    # game_list = ["zero_sum", "general_sum"]

    psro(generator=generator,
         game_type=FLAGS.game_type,
         num_rounds=FLAGS.num_rounds,
         seed=seed,
         checkpoint_dir=checkpoint_dir,
         num_iterations=FLAGS.num_iterations)


if __name__ == "__main__":
  app.run(main)
