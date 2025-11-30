import pandas as pd
import sys
import yaml

from data_prep import preprocess_csv
from data_prep import undo_preprocess
from optimize import run_optimizer

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Data
    df, one_hot = preprocess_csv(config['path_to_data'], sep=',')

    # Run the optimizer on the data, get the synthetic data
    sf = run_optimizer(df, one_hot, config['target_columns'], evals=config['optimization_passes'], n_samples=config['sample_size'])
    
    # Undo data preprocessing, return encodings to categorical
    sf = undo_preprocess(sf)

    # Write data to csv
    sf.to_csv(config['synthetic_data_path'], index=False)