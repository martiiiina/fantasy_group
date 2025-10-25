import numpy as np

# 1. Cramér’s V for correlation between categorical variables
def cramers_v_numpy(x, y):
    """Calcola Cramér’s V solo con NumPy"""
    # Costruisci tabella di contingenza
    categories_x = np.unique(x[~np.isnan(x)])
    categories_y = np.unique(y[~np.isnan(y)])
    n = len(x)

    # Tabella di contingenza
    contingency = np.zeros((len(categories_x), len(categories_y)))
    for i, val_x in enumerate(categories_x):
        for j, val_y in enumerate(categories_y):
            contingency[i, j] = np.sum((x == val_x) & (y == val_y))

    # Calcola chi-quadro
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / contingency.sum()
    chi2 = np.nansum((contingency - expected)**2 / expected)

    # Cramér’s V
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / max(1e-10, min((kcorr - 1), (rcorr - 1))))  # evita div 0

# 2. Correlation matrix
def compute_correlation_matrix(x_train, categorical_idx, continuous_idx, threshold=0.9):
    n_features = x_train.shape[1]
    corr_matrix = np.zeros((n_features, n_features)) * np.nan  # riempi con NaN

    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue

            # Caso numerico-numerico → Pearson
            if i in continuous_idx and j in continuous_idx:
                xi = x_train[:, i]
                xj = x_train[:, j]
                mask = ~np.isnan(xi) & ~np.isnan(xj)
                if np.sum(mask) > 1:
                    corr_ij = np.corrcoef(xi[mask], xj[mask])[0, 1]
                else:
                    corr_ij = np.nan

            # Caso categorico-categorico → Cramér’s V
            elif i in categorical_idx and j in categorical_idx:
                xi = x_train[:, i]
                xj = x_train[:, j]
                corr_ij = cramers_v_numpy(xi, xj)

            # Caso misto → non confrontabile
            else:
                corr_ij = np.nan

            corr_matrix[i, j] = corr_ij
            corr_matrix[j, i] = corr_ij

    return corr_matrix

# 3. Remove redundant features: Cramer's V for categorical, correlation for numerical
def drop_redundant_features(x_train, x_test, corr_matrix, threshold=0.9):
    non_valid_col = set()
    n_features = x_train.shape[1]

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.isnan(corr_matrix[i, j]):
                continue
            if abs(corr_matrix[i, j]) > threshold:
                non_valid_col.add(j)  # tieni i, droppa j

    x_train_decorr = np.delete(x_train, list(non_valid_col), axis=1)
    x_test_decorr = np.delete(x_test, list(non_valid_col), axis=1)
    return x_train_decorr, x_test_decorr, list(non_valid_col)

# 4. Entropy
def entropy(arr):
    vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

# 5. Mutual information

def mutual_information(x, y, bins=10):
    # Discretizza se continua
    if len(np.unique(x)) > bins:
        x = np.digitize(x, np.histogram(x[~np.isnan(x)], bins=bins)[1])
    if len(np.unique(y)) > bins:
        y = np.digitize(y, np.histogram(y[~np.isnan(y)], bins=bins)[1])
    h_x = entropy(x)
    h_y = entropy(y)
    h_xy = entropy(np.array(list(zip(x, y))))
    return h_x + h_y - h_xy