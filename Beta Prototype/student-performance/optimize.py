from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO

from vae import generate_data
from run_tests import get_optimization_score

problem = HpProblem()
problem.add_hyperparameter((4, 64), "LATENT_DIM")
problem.add_hyperparameter((10, 100), "EPOCHS")
problem.add_hyperparameter((10, 50), "BATCH_SIZE")

def run(config):
    generate_data(**config)
    return get_optimization_score()

if __name__ == "__main__":
    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator)
    results = search.search(max_evals=30)