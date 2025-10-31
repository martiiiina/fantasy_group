def main():
    # Import libraries
    from helpers import load_csv_data
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from support import remove_outliers_categorical, corrcoef, one_hot_encode_columns, apply_one_hot_encoding, split_categorical_continuous

    # Load data, check dimensions and data type
    print("Loading data...")
    data_path='data/dataset/dataset'
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)
    print("Number of samples of train: ", x_train.shape[0])
    print("Number of features: ", x_train.shape[1])
    print("Number of samples of test: ", x_test.shape[0])
    print("Data type x_train:", x_train.dtype) 
    print("Data type y_train:", y_train.dtype) 

    # Types of variables (continuous and categorical) based on the number of unique values
    threshold = 20                                      # Fixed threshold
    categorical_idx, continuous_idx, unique_counts = split_categorical_continuous(x_train, threshold=20)

    print(f"Total categorical features (<= {threshold} unique): {len(categorical_idx)}")
    print(f"Total continuous features    (>  {threshold} unique): {len(continuous_idx)}")

    unique_counts = np.array(unique_counts)
    unique_vals, counts = np.unique(unique_counts, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique_vals, counts, color='lightblue', width=1.0)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Number of unique values')
    plt.xlim(right=200)
    plt.xlim(left=0)
    plt.ylabel('Number of features')
    plt.title('Distribution of unique values per feature')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Data cleaning: remove non valid values from categorical features (e.g. 1 2 3 9 --> 9 = non valid)
    thresh_quant=0.9
    x_train_nan = remove_outliers_categorical(x_train,categorical_idx, thresh_quant)
    x_test_nan = remove_outliers_categorical(x_test,categorical_idx, thresh_quant)

    # Feature selection: remove features with >40% NaN values 
    nan_ratio = np.mean(np.isnan(x_train_nan), axis=0)
    valid_cols = np.where(nan_ratio < 0.4)[0]
    x_train_clean = x_train_nan[:, valid_cols]
    x_test_clean = x_test_nan[:, valid_cols]
    print(f"Number of features after removal of >40% NaN features: {x_train_clean.shape[1]}")

    # Impute remaining NaN with median (continuous) and -1 (categorical)
    categorical_idx, continuous_idx, _ = split_categorical_continuous(x_train_clean, threshold=20)
    for i in continuous_idx:
        median = np.nanmedian(x_train_clean[:, i])
        x_train_clean[np.isnan(x_train_clean[:, i]), i] = median
        x_test_clean[np.isnan(x_test_clean[:, i]), i] = median
    for i in categorical_idx:
        x_train_clean[np.isnan(x_train_clean[:, i]), i] = - 1
        x_test_clean[np.isnan(x_test_clean[:, i]), i] = - 1
    print("Missing values imputed (median for continuous, -1 for categorical)")

    # Feature selection: remove highly correlated features, drop features if correlation > 0.9
    corr_matrix = corrcoef(x_train_clean)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation matrix before cleaning')
    plt.show()
    non_valid_col = set()  # Use set to avoid duplicates
    for i in range(x_train_clean.shape[1]):
        for j in range(i + 1, x_train_clean.shape[1]):
            if abs(corr_matrix[i, j]) > 0.9:
                non_valid_col.add(j)  # Drop j, keep i
    x_train_decorr = np.delete(x_train_clean, list(non_valid_col), axis=1)     # Remove column
    x_test_decorr = np.delete(x_test_clean, list(non_valid_col), axis=1)
    print(f"Number of features after decorrelation: {x_train_decorr.shape[1]}")

    # One-hot encoding for categorical features 
    categorical_idx, continuous_idx, _ = split_categorical_continuous(x_train_decorr, threshold=20)
    X_train_encoded, category_map =  one_hot_encode_columns(x_train_decorr, categorical_idx)
    X_test_encoded = apply_one_hot_encoding(x_test_decorr, categorical_idx, category_map)
    print("Number of features after one-hot encoding: ", X_train_encoded.shape[1])

    # Recover indexes of continuous columns after one-hot encoding
    n_features_original = x_train_decorr.shape[1]           
    continuous_mask = np.ones(n_features_original, dtype=bool)
    continuous_mask[categorical_idx] = False
    new_continuous_idx = []
    col_counter = 0
    for i in range(n_features_original):
        if continuous_mask[i]:
            # Continuous features --> add one column
            new_continuous_idx.append(col_counter)
            col_counter += 1
        else:
            # Categorical features --> as many colmuns as categories
            n_values = len(category_map[i])
            col_counter += n_values
    print(f"Number of continuous features after one-hot encoding: {len(new_continuous_idx)}")

    # Z-score normalization on continuous features
    col_mean = np.mean(X_train_encoded[:, new_continuous_idx], axis=0)
    col_std = np.std(X_train_encoded[:, new_continuous_idx], axis=0)

    x_train_norm = X_train_encoded.copy()
    x_train_norm[:, new_continuous_idx] = (X_train_encoded[:, new_continuous_idx] - col_mean) / col_std
    x_test_norm = X_test_encoded.copy()
    x_test_norm[:, new_continuous_idx] = (X_test_encoded[:, new_continuous_idx] - col_mean) / col_std

    # Check labels distribution
    plt.hist(y_train, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribuzione delle etichette (y_train)")
    plt.xlabel("Valore")
    plt.ylabel("Frequenza")
    plt.show()

    # Save processed data
    np.save("processed/x_train.npy", x_train_norm)
    np.save("processed/x_test.npy", x_test_norm)
    np.save("processed/y_train.npy", y_train)
    np.save("processed/train_ids.npy", train_ids)
    np.save("processed/test_ids.npy", test_ids)

if __name__ == "__main__":
    main()
