import pandas as pd
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO

from vae import generate_data
from scoring import get_optimization_score

problem = HpProblem()
problem.add_hyperparameter((4, 24), "LATENT_DIM")
problem.add_hyperparameter((10, 100), "EPOCHS")
problem.add_hyperparameter((0.001, 0.05), "kl_weight")

g_df = None
g_sf = None
g_one_hot = None

def run(config):
    global g_df
    global g_sf
    global g_one_hot

    print(f'New test:')
    g_sf = generate_data(g_df, g_one_hot, **config)
    score = get_optimization_score(g_df, g_sf)
    print()
    return score

def run_optimizer(df, one_hot, evals=30):
    global g_df
    global g_sf
    global g_one_hot

    g_df = df
    g_one_hot = one_hot

    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator, log_dor="./logs")
    results = search.search(max_evals=evals)

    print(g_sf)

    return g_sf

if __name__ == "__main__":
    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator, log_dir="./logs")
    results = search.search(max_evals=30)

    print(results).preproce