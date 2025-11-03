from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO

from vae import generate_data
from scoring import get_optimization_score

problem = HpProblem()
problem.add_hyperparameter((4, 24), "LATENT_DIM")
problem.add_hyperparameter((10, 100), "EPOCHS")
problem.add_hyperparameter((0.001, 0.05), "kl_weight")

def run(config):
    print(f'New test:')
    generate_data(**config)
    score = get_optimization_score()
    print()
    return score

if __name__ == "__main__":
    evaluator = Evaluator.create(run, method="thread")
    search = CBO(problem, evaluator)
    results = search.search(max_evals=30)