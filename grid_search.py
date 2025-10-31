import os
from itertools import product
import csv
import preprocessing
import Training

# Temporary directory
os.makedirs("tmp", exist_ok=True)

# Hyperparameters grid
thresh_nan = [0.35, 0.4, 0.45]
thresh_quant = [0.9, 0.95, 0.99]
thresh_corr = [0.8, 0.85, 0.9]
gammas = [0.0001, 0.0005, 0.001, 0.01, 0.1]
batch_sizes = [64, 128, 256]
nums_batches = [23, 30, 50]
lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]

results = []
results_path_csv = "results.csv"


# Loop Preprocessing + Training 
for thr_n, thr_q, thr_c in product(thresh_nan, thresh_quant, thresh_corr):
    print(f"\n=== Preprocessing: NaN={thr_n}, Quant={thr_q}, Corr={thr_c} ===")

    # Run preprocessing.py
    preprocessing.run(
        thresh_nan=thr_n,
        thresh_quant=thr_q,
        thresh_corr=thr_c
    )

    # Loop training
    for gamma, batch_size, num_batches, lambda_ in product(gammas, batch_sizes, nums_batches, lambdas):
        print(f"Training γ={gamma} | batch_size={batch_size} | num_batches={num_batches} | lambda={lambda_}")

        gen_err = Training.run(
            gamma=gamma,
            batch_size=batch_size,
            num_batches=num_batches,
            lambda_=lambda_
        )

        # Result
        result_row = {
            "thresh_nan": thr_n,
            "thresh_quant": thr_q,
            "thresh_corr": thr_c,
            "gamma": gamma,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "lambda": lambda_,
            "gen_err": gen_err
        }

        results.append(result_row)

        # Save
        file_exists = os.path.exists(results_path_csv)
        with open(results_path_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_row)

        print(f"Saved result: gen_err={gen_err:.5f}")
