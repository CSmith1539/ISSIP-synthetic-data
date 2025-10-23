from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator


def run(job):
    x = job.parameters["x"]
    b = job.parameters["b"]
    function = job.parameters["function"]

    if function == "linear":
        y = x + b
    elif function == "cubic":
        y = x**3 + b

    return y


def optimize():
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")
    problem.add_hyperparameter((0, 10), "b")
    problem.add_hyperparameter(["linear", "cubic"], "function")

    evaluator = Evaluator.create(run, method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    search = CBO(
        problem, 
        random_state=42, 
        solution_selection="argmax_obs",
    )
    results = search.search(evaluator, max_evals=100)

    return results

def test_opt():
    return 5

if __name__ == "__main__":
    results = optimize()
    print(results)

    row = results.iloc[-1]
    print("\nOptimum values")
    print("function:", row["sol.p:function"])
    print("x:", row["sol.p:x"])
    print("b:", row["sol.p:b"])
    print("y:", row["sol.objective"])