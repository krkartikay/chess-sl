from experiment import Experiment

import train
import gen_moves


def run_exp(exp):
    # num_games = exp.get_config("num_games")
    n_hidden = exp.get_config("n_hidden")
    batch_size = exp.get_config("batch_size")
    learning_rate = exp.get_config("learning_rate")
    num_epochs = exp.get_config("num_epochs")
    weight_decay = exp.get_config("weight_decay")
    # gen_moves.generate_games(num_games)
    return train.run_training(exp, n_hidden, batch_size, learning_rate, num_epochs, weight_decay)


exp = Experiment(run_exp, name="weight_decay")

exp.add_config("num_epochs", [20, 100], dev_run_value=20)
exp.add_config("num_games", [100, 500], dev_run_value=100)
exp.add_config("n_hidden", [512, 2048, 4096], dev_run_value=64)
exp.add_config("batch_size", [8, 64, 256, 512], dev_run_value=64)
exp.add_config("learning_rate", [0.1, 1, 10])
exp.add_config("weight_decay", [1e-3, 1e-2, 1e-1])
exp.add_config("dummy_run_num", [1])

exp.run_experiment_full()
# exp.run_experiment_single_variable()
