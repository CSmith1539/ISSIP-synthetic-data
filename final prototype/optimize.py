import pandas as pd
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO

from vae import generate_data
from scoring import get_optimization_score

problem = HpProblem()
problem.add_hyperparameter((8, 32), "LATENT_DIM")
problem.add_hyperparameter((10, 100), "EPOCHS")
problem.add_hyperparameter((0.00001, 0.1, "log-uniform"), "kl_weight")

g_df = None
g_sf = None
g_one_hot = None
g_n_samples = None
g_target_columns = None

g_tests = 1

def run(config):
    global g_df
    global g_sf
    global g_one_hot
    global g_target_columns
    global g_tests

    print(f'New test: ', g_tests)
    g_sf = generate_data(g_df, g_one_hot, g_n_samples, **config)
    score = get_optimization_score(g_df, g_sf, g_target_columns)
    g_tests = g_tests + 1
    print()
    return score

def run_optimizer(df, one_hot, target_columns, evals=30, n_samples=1000):
    global g_df
    global g_sf
    global g_one_hot
    global g_n_samples
    global g_target_columns

    g_df = df
    g_one_hot = one_hot
    g_n_samples = n_samples
    g_target_columns = target_columns

    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator, log_dir="./logs")
    results = search.search(max_evals=evals)

    return g_sf

if __name__ == "__main__":
    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator, log_dir="./logs")
    results = search.search(max_evals=30)

    print(results).preproce