from meta_strategies import double_oracle, fictitious_play
from game_generator import Game_generator
from psro_trainer import PSRO_trainer

from absl import app
from absl import flags
import os
import pickle
import datetime

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_rounds", 1, "The number of rounds starting with different.")
flags.DEFINE_integer("num_strategies", 10, "The number of rounds starting with different.")

def psro(generator, game_type, num_rounds, checkpoint_dir, meta_method_list=None, blocks=False):
    if game_type == "zero_sum":
        meta_games = generator.zero_sum_game()
    elif game_type == "general_sum":
        meta_games = generator.general_sum_game()
    elif game_type == "symmetric_zero_sum":
        meta_games = generator.general_sum_game()
    else:
        raise ValueError

    DO_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=double_oracle,
                           checkpoint_dir=checkpoint_dir,
                           meta_method_list=meta_method_list,
                           blocks=blocks)

    FP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=fictitious_play,
                           checkpoint_dir=checkpoint_dir,
                           meta_method_list=meta_method_list,
                           blocks=blocks)

    DO_FP_trainer = PSRO_trainer(meta_games=meta_games,
                              num_strategies=generator.num_strategies,
                              num_rounds=num_rounds,
                              meta_method=double_oracle,
                              checkpoint_dir=checkpoint_dir,
                              meta_method_list=[double_oracle, fictitious_play],
                              blocks=blocks)

    DO_trainer.loop()
    FP_trainer.loop()
    DO_FP_trainer.loop()

    print("The current game type is ", game_type)
    print("DO number of iters:", DO_trainer.num_used_iterations)
    print("FP number of iters:", FP_trainer.num_used_iterations)
    print("DO+FP number of iters:", DO_FP_trainer.num_used_iterations)
    print("====================================================")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + game_type + '_meta_games.pkl','wb') as f:
        pickle.dump(meta_games, f)
    with open(checkpoint_dir + game_type + '_DO.pkl','wb') as f:
        pickle.dump(DO_trainer.num_used_iterations, f)
    with open(checkpoint_dir + game_type + '_FP.pkl','wb') as f:
        pickle.dump(FP_trainer.num_used_iterations, f)
    with open(checkpoint_dir + game_type + '_DO_SP.pkl','wb') as f:
        pickle.dump(DO_FP_trainer.num_used_iterations, f)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    generator = Game_generator(FLAGS.num_strategies)
    checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir) + '/'

    game_list = ["zero_sum", "general_sum", "symmetric_zero_sum"]

    for game in game_list:
        psro(generator=generator,
             game_type=game,
             num_rounds=FLAGS.num_rounds,
             checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
  app.run(main)



