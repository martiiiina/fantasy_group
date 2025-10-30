import os
import papermill as pm
import pandas as pd
from itertools import product
import json

# --- Creazione cartella temporanea per Papermill ---
os.makedirs("tmp", exist_ok=True)

# --- Griglie di iperparametri ---
thresh_nan = [0.4]
thresh_quant = [0.9]
thresh_corr = [0.9]
gammas = [0.0001, 0.0005, 0.001, 0.01, 0.1]
batch_sizes = [64, 128, 256]
nums_batches = [23, 30, 50]

results = []
results_path_csv = "results.csv"
results_path_xlsx = "results.xlsx"

# --- Se esiste già un file di risultati, lo carica per continuare ---
if os.path.exists(results_path_csv):
    df_existing = pd.read_csv(results_path_csv)
    results = df_existing.to_dict(orient="records")
    print(f"📄 Ripreso da {len(results)} risultati precedenti.")
else:
    print("🆕 Nessun file risultati trovato, inizio nuova sessione.")

# --- Loop EDA + Training ---
for thr_n, thr_q, thr_c in product(thresh_nan, thresh_quant, thresh_corr):
    print(f"\n=== EDA: NaN={thr_n}, Quant={thr_q}, Corr={thr_c} ===")

    # 1️⃣ Esegui EDA (salva sempre gli stessi file .npy)
    pm.execute_notebook(
        'eda1.ipynb',
        'tmp/tmp_eda_output.ipynb',
        parameters=dict(
            thresh_nan=thr_n,
            thresh_quant=thr_q,
            thresh_corr=thr_c
        )
    )

    # 2️⃣ Loop training
    for gamma, batch_size, num_batches  in product(gammas, batch_sizes, nums_batches):
        print(f"▶ Train γ={gamma} | batch_size={batch_size} | num_batches={num_batches}")

        nb_result = pm.execute_notebook(
            'train.ipynb',
            'tmp/tmp_train_output.ipynb',
            parameters=dict(
                gamma=gamma,
                batch_size=batch_size,
                num_batches=num_batches
            )
        )

        # Leggi il valore di gen_err salvato nel file JSON
        with open("tmp/results.json") as f:
            gen_err = json.load(f)["gen_err"]

        # Aggiungi il risultato corrente
        result_row = {
            "thresh_nan": thr_n,
            "thresh_quant": thr_q,
            "thresh_corr": thr_c,
            "gamma": gamma,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "gen_err": gen_err
        }
        results.append(result_row)

        # --- Salvataggio progressivo ---
        df = pd.DataFrame(results)
        df.to_csv(results_path_csv, index=False)
        df.to_excel(results_path_xlsx, index=False)

        print(f"💾 Risultato salvato: gen_err={gen_err:.5f}")

print("\n✅ Tutte le configurazioni completate!")
print(f"📊 Risultati finali salvati in {results_path_csv} e {results_path_xlsx}")
