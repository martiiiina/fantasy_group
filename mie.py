import numpy as np
from implementations import logistic_regression, reg_logistic_regression
from common import sigmoid, batch_iter

def split_categorical_continuous(X, threshold=20):
    categorical_idx = []
    continuous_idx = []
    unique_counts = []
    for i in range(X.shape[1]):
        count = len(np.unique(X[:, i]))
        (categorical_idx if count <= threshold else continuous_idx).append(i)
        unique_counts.append(count)
    return categorical_idx, continuous_idx, unique_counts

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)            # n° cols output (i.e. number of items for each fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)    # all possible indices in y, shuffled
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]  # slicing all possible indices in k folds from 0 to k_fold
    return np.array(k_indices)                 # returns shuffled indices grouped in k rows of n/k elements

def logistic_loss(y, tx, w):
    N=y.shape[0]
    sig = sigmoid(tx @ w)
    loss = -(1/N) * (y.T @ np.log(sig) + (1-y).T @ np.log(1-sig))
    loss=np.squeeze(loss).item()
    return loss

def subsample_class(x, y, majority_class=0, target_ratio=1.0, seed=42):
    np.random.seed(seed)
    majority_mask = (y == majority_class)
    minority_mask = ~majority_mask

    x_majority = x[majority_mask, :]
    y_majority = y[majority_mask]
    x_minority = x[minority_mask, :]
    y_minority = y[minority_mask]
    
    n_minority = x_minority.shape[0]
    n_majority_sample = int(n_minority * target_ratio)

    indices = np.random.choice(x_majority.shape[0], n_majority_sample, replace=False) # selection with no repetition
    
    x_majority_sampled = x_majority[indices, :]
    y_majority_sampled = y_majority[indices]
    
    x_balanced = np.vstack((x_majority_sampled, x_minority))
    y_balanced = np.concatenate((y_majority_sampled, y_minority))
    
    # Shuffle on x and y
    perm = np.random.permutation(x_balanced.shape[0])
    return x_balanced[perm], y_balanced[perm]

import numpy as np

def subsample_class_methods(X, y, target_ratio=1.0, method="random", n_iter=10):
    """
    Bilancia le classi del dataset utilizzando vari metodi di subsampling.

    Parametri
    ----------
    X : np.ndarray
        Matrice delle feature (n_samples, n_features)
    y : np.ndarray
        Vettore delle etichette (valori -1 e 1)
    target_ratio : float
        Rapporto desiderato tra minoranza e maggioranza (1.0 = bilanciato)
    method : str
        Metodo di bilanciamento:
        - "random"     : sottocampionamento casuale
        - "cluster"    : sottocampionamento basato su cluster (mini K-means)
        - "nearmiss"   : sottocampionamento dei punti più vicini alla minoranza
        - "tomek"      : rimozione dei Tomek Links (data cleaning)
        - "cnn"        : Condensed Nearest Neighbor (mantiene punti rilevanti)
    n_iter : int
        Numero di iterazioni per metodi iterativi (es. cluster, cnn)

    Ritorna
    -------
    X_new, y_new : np.ndarray
        Dati bilanciati secondo il metodo scelto
    """

    # Separa classi
    X_pos, X_neg = X[y == 1], X[y == -1]
    n_pos, n_neg = len(X_pos), len(X_neg)

    # Identifica maggioranza/minoranza
    if n_pos > n_neg:
        X_major, y_major, X_minor, y_minor = X_pos, np.ones(n_pos), X_neg, -np.ones(n_neg)
    else:
        X_major, y_major, X_minor, y_minor = X_neg, -np.ones(n_neg), X_pos, np.ones(n_pos)

    n_target = int(len(X_minor) * target_ratio)
    n_target = min(n_target, len(X_major))  # Evita errori

    # === RANDOM UNDERSAMPLING ===
    if method == "random":
        idx = np.random.choice(len(X_major), n_target, replace=False)
        X_major_sel = X_major[idx]

    # === CLUSTER UNDERSAMPLING (mini K-means) ===
    elif method == "cluster":
        centroids = X_major[np.random.choice(len(X_major), n_target, replace=False)]
        for _ in range(n_iter):
            dist = np.linalg.norm(X_major[:, None] - centroids[None, :], axis=2)
            closest = np.argmin(dist, axis=1)
            for j in range(n_target):
                pts = X_major[closest == j]
                if len(pts) > 0:
                    centroids[j] = pts.mean(axis=0)
        X_major_sel = centroids

    # === NEARMISS UNDERSAMPLING ===
    elif method == "nearmiss":
        distances = np.linalg.norm(X_major[:, None] - X_minor[None, :], axis=2)
        mean_dist = distances.mean(axis=1)
        idx = np.argsort(mean_dist)[:n_target]
        X_major_sel = X_major[idx]

    # === TOMEK LINKS CLEANING ===
    elif method == "tomek":
        X_new, y_new = _tomek_links(X, y)
        return X_new, y_new

    # === CONDENSED NEAREST NEIGHBOR ===
    elif method == "cnn":
        X_new, y_new = _condensed_nn(X, y, max_iter=n_iter)
        return X_new, y_new

    else:
        raise ValueError(f"Metodo '{method}' non riconosciuto. Usa: random, cluster, nearmiss, tomek, cnn.")

    # Combina minoranza + campioni selezionati della maggioranza
    y_major_sel = np.full(len(X_major_sel), y_major[0])
    X_new = np.vstack((X_minor, X_major_sel))
    y_new = np.hstack((y_minor, y_major_sel))

    # Shuffle finale per evitare bias d'ordine
    idx = np.random.permutation(len(X_new))
    return X_new[idx], y_new[idx]


# === SUPPORT FUNCTIONS ===

def _tomek_links(X, y):
    """Rimuove i Tomek Links per migliorare la separabilità delle classi."""
    X_pos, X_neg = X[y == 1], X[y == -1]
    keep_idx = np.ones(len(X), dtype=bool)

    for i, x_p in enumerate(X_pos):
        dist = np.linalg.norm(X_neg - x_p, axis=1)
        j = np.argmin(dist)
        x_n = X_neg[j]
        dist2 = np.linalg.norm(X_pos - x_n, axis=1)
        if np.argmin(dist2) == i:
            idx_n = np.where((X == x_n).all(axis=1))[0][0]
            keep_idx[idx_n] = False
    return X[keep_idx], y[keep_idx]


def _condensed_nn(X, y, max_iter=10):
    """Implementa il Condensed Nearest Neighbor (CNN) per ridurre la ridondanza."""
    idx = np.random.choice(len(X), size=1)
    X_store, y_store = X[idx], y[idx]

    for _ in range(max_iter):
        changed = False
        for xi, yi in zip(X, y):
            dist = np.linalg.norm(X_store - xi, axis=1)
            nearest = np.argmin(dist)
            if y_store[nearest] != yi:
                X_store = np.vstack((X_store, xi))
                y_store = np.hstack((y_store, yi))
                changed = True
        if not changed:
            break
    return X_store, y_store



def cross_validation(y, x, k_indices, k, initial_w, max_iters, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """
    # IDEA: at every k-th iteration I select the k-th row of k_indices as the test indices,
    # and all the other rows as the training indices

    val_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]      # list from 0 to K_fold, == current k, list of zeros and one 1 for current k, inverted with tilde
    
    # N.B. passing a boolean to k_indices like T T F T T, slicing of rows of k_indices with index True

    tr_indice = tr_indice.reshape(-1)   # trasforms tr_indice from matrix to 1D

    y_val = y[val_indice]
    y_tr = y[tr_indice]
    x_val = x[val_indice, :]
    x_tr = x[tr_indice, :]

    w, loss_tr, loss_val = logistic_regression(y_tr, x_tr, y_val, x_val, initial_w, max_iters, gamma)

    return w, loss_tr, loss_val, x_val, y_val

def train_val_split(x, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

