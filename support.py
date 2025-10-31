import numpy as np

def remove_outliers_categorical(X,  categorical_idx, threshold_q):
    """
    Replace anomalous or extreme values with np.nan
    threshold = limit quantile (e.g. 0.90 = over the 90° percentile)
    """
    X_clean = X.copy().astype(float)
    for i in range(X.shape[1]):
        if i not in categorical_idx: 
             continue
        col = X[:, i]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) == 0:
            continue
        max_val = np.percentile(col_valid, threshold_q * 100)
        X_clean[col > max_val, i] = np.nan
    return X_clean

def corrcoef(X):
    """
    Computes correlation matrix between two features of X
    """
    n_features = X.shape[1]
    corr = np.empty((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            corr_ij = np.corrcoef(X[:, i], X[:, j])[0, 1]       # np.corrcoef returns correlation matrix of two 1D-arrays, corr_ij is in position [0,1]            
            corr[i, j] = corr_ij
            corr[j, i] = corr_ij  
    return corr

def one_hot_encode_columns(X, categorical_cols):
    """
    Applies one-hot encoding on categorical columns of x_train.
    Returns the new array and a dictionary with used categories.
    """
    X_encoded = []
    category_map = {}  # to remind which categories has every column

    for i in range(X.shape[1]):
        if i in categorical_cols:
            # unique values in the column
            values = np.unique(X[:, i])
            category_map[i] = values

            # Creates a one-hot matrix for each value
            one_hot = np.zeros((X.shape[0], len(values)))
            for j, val in enumerate(values):
                one_hot[:, j] = (X[:, i] == val).astype(float)
            X_encoded.append(one_hot)
        else:
            # leaves the numerical column unchanged
            X_encoded.append(X[:, i].astype(float).reshape(-1, 1))
    
    # final concatenation of all columns
    X_encoded = np.concatenate(X_encoded, axis=1)
    return X_encoded, category_map


def apply_one_hot_encoding(X, categorical_cols, category_map):
    """
    Applies the encoding to the test set using categories already found on the training set.
    If a category is not present in the training, it is ignored (0 column).
    """
    X_encoded = []

    for i in range(X.shape[1]):
        if i in categorical_cols:
            values = category_map[i]
            one_hot = np.zeros((X.shape[0], len(values)))
            for j, val in enumerate(values):
                one_hot[:, j] = (X[:, i] == val).astype(float)
            X_encoded.append(one_hot)
        else:
            X_encoded.append(X[:, i].astype(float).reshape(-1, 1))

    X_encoded = np.concatenate(X_encoded, axis=1)
    return X_encoded

def logistic_regression(y_tr, x_tr, y_val, x_val, initial_w, max_iters, gamma):
    """
    Simple logistic regression (no regularization)
    """
    w = initial_w
    losses = []
    losses_val = []

    y_tr = y_tr.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    for iter in range(max_iters):
        # Predictions
        N = y_tr.shape[0]
        sig = sigmoid(x_tr @ w)
        # Gradient update
        grad = (1 / N) * x_tr.T @ (sig - y_tr)
        w = w - gamma * grad
        # Loss
        sig = sigmoid(x_tr @ w)
        loss = -(1 / N) * (y_tr.T @ np.log(sig) + (1 - y_tr).T @ np.log(1 - sig))
        loss = np.squeeze(loss)
        losses.append(loss)

        # Validation
        sig_val = sigmoid(x_val @ w)
        N_val = y_val.shape[0]
        loss_val = -(1 / N_val) * (y_val.T @ np.log(sig_val) + (1 - y_val).T @ np.log(1 - sig_val))
        loss_val = np.squeeze(loss_val)
        losses_val.append(loss_val)
    return w, losses, losses_val


def reg_logistic_regression(y_tr, x_tr, y_val, x_val, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression
    """
    w = initial_w
    losses = []
    losses_val = []

    y_tr = y_tr.reshape(-1, 1)  # from (N,) to (N,1)
    y_val = y_val.reshape(-1, 1)

    for iter in range(max_iters):
        # Predictions
        N = y_tr.shape[0]
        sig = sigmoid(x_tr @ w)
        sig = np.clip(sig, 1e-15, 1 - 1e-15)
        # Gradient update
        grad = (1/N) * x_tr.T @ (sig - y_tr) + 2 * lambda_ * w
        w = w - gamma * grad
        # Loss
        sig = sigmoid(x_tr @ w)
        sig = np.clip(sig, 1e-15, 1 - 1e-15)
        loss = -(1 / N) * (y_tr.T @ np.log(sig) + (1 - y_tr).T @ np.log(1 - sig))
        loss = np.squeeze(loss)
        losses.append(loss)

        # Validation
        sig_val = sigmoid(x_val @ w)
        sig_val = np.clip(sig_val, 1e-15, 1 - 1e-15)
        N_val = y_val.shape[0]
        loss_val = -(1 / N_val) * (y_val.T @ np.log(sig_val) + (1 - y_val).T @ np.log(1 - sig_val))
        loss_val = np.squeeze(loss_val)
        losses_val.append(loss_val)
    
    return w, losses, losses_val

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def split_categorical_continuous(X, threshold=20):
    """
    Split cateogrical and continuous features based on the number of unique values
    """
    categorical_idx = []
    continuous_idx = []
    unique_counts = []
    for i in range(X.shape[1]):
        count = len(np.unique(X[:, i]))
        (categorical_idx if count <= threshold else continuous_idx).append(i)
        unique_counts.append(count)
    return categorical_idx, continuous_idx, unique_counts


def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(
        num_row / k_fold
    )  # n° cols output (i.e. number of items for each fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)  # all possible indices in y, shuffled
    k_indices = [
        indices[k * interval : (k + 1) * interval] for k in range(k_fold)
    ]  # slicing all possible indices in k folds from 0 to k_fold
    return np.array(
        k_indices
    )  # returns shuffled indices grouped in k rows of n/k elements

def subsample_class(x, y, majority_class=0, target_ratio=1.0, seed=42):
    """
    Performs undersampling of majoritary class
    """
    np.random.seed(seed)
    majority_mask = y == majority_class
    minority_mask = ~majority_mask

    x_majority = x[majority_mask, :]
    y_majority = y[majority_mask]
    x_minority = x[minority_mask, :]
    y_minority = y[minority_mask]

    n_minority = x_minority.shape[0]
    n_majority_sample = int(n_minority * target_ratio)

    indices = np.random.choice(
        x_majority.shape[0], n_majority_sample, replace=False
    )  # selection with no repetition

    x_majority_sampled = x_majority[indices, :]
    y_majority_sampled = y_majority[indices]

    x_balanced = np.vstack((x_majority_sampled, x_minority))
    y_balanced = np.concatenate((y_majority_sampled, y_minority))

    # Shuffle on x and y
    perm = np.random.permutation(x_balanced.shape[0])
    return x_balanced[perm], y_balanced[perm]


def cross_validation(y,x,k_indices,k,initial_w,max_iters,gamma,lambda_):
    """
    Performs cross validation on x
    At every k-th iteration it selects the k-th row of k_indices as the test indices,
    and all the other rows as the training indices
    """
    
    val_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]  

    tr_indice = tr_indice.reshape(-1)  

    y_val = y[val_indice]
    y_tr = y[tr_indice]
    x_val = x[val_indice, :]
    x_tr = x[tr_indice, :]

    w, loss_tr, loss_val = reg_logistic_regression(y_tr, x_tr, y_val, x_val, lambda_, initial_w, max_iters, gamma)

    return w, loss_tr, loss_val


def logistic_regression_with_Adam(y,tx,y_val,x_val,initial_w,initial_m,initial_v,beta1,beta2,
                                  batch_size,num_batches,max_iters,gamma):
    """
    Logistic regression with Adam scheduler
    """
    w = initial_w.reshape(-1, 1)
    m = initial_m.reshape(-1, 1)
    v = initial_v.reshape(-1, 1)
    losses = []
    losses_val = []

    y = y.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    eps = 1e-15  # to prevent log(0)

    grads = []
    for iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(
            y, tx, batch_size=batch_size, num_batches=num_batches, shuffle=True):  
            sig = sigmoid(batch_tx @ w)
            sig = np.clip(sig, 1e-15, 1 - 1e-15)
            N = batch_y.shape[0]
            grad = (1 / N) * batch_tx.T @ (sig - batch_y)
            grads.append(grad)
        stoch_grad = np.mean(grads, axis=0)
        N = y.shape[0]
        sig = sigmoid(tx @ w)
        sig = np.clip(sig, 1e-15, 1 - 1e-15)
        loss = -(1 / N) * (y.T @ np.log(sig) + (1 - y).T @ np.log(1 - sig))
        loss = np.squeeze(loss)
        losses.append(loss)

        # Validation
        sig_val = sigmoid(x_val @ w)
        sig_val = np.clip(sig_val, 1e-15, 1 - 1e-15)
        N_val = y_val.shape[0]
        loss_val = -(1 / N_val) * (y_val.T @ np.log(sig_val) + (1 - y_val).T @ np.log(1 - sig_val))
        loss_val = np.squeeze(loss_val)
        losses_val.append(loss_val)

        t = iter + 1
        m = beta1 * m + (1 - beta1) * stoch_grad
        v = beta2 * v + (1 - beta2) * stoch_grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        w = w - gamma * m_hat / (np.sqrt(v_hat) + eps)
    return w, losses, losses_val
