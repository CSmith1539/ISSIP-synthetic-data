import pandas as pd
import sys

from data_prep import preprocess_csv
from optimize import run_optimizer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please run "python3 run.py /path/to/dataset')
    else:
        # Load Data
        df, one_hot = preprocess_csv(sys.argv[1], sep=';')

        # Run the optimizer on the data, get the synthetic data
        sf = run_optimizer(df, one_hot, evals=1)

        # Write data to csv
        sf.to_csv('output/synthetic_data.csv')
