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

def cross_validation(y, x, k_indices, k, initial_w, initial_m, initial_v, beta1, beta2, batch_size, num_batches, max_iters, gamma):
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

    w, loss_tr, loss_val = logistic_regression_with_Adam(y_tr, x_tr, y_val, x_val, initial_w, initial_m, initial_v, beta1, beta2, batch_size, num_batches, max_iters, gamma)

    return w, loss_tr, loss_val, x_val, y_val

def train_val_split(x, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def logistic_regression_with_Adam(y, tx, y_val, x_val, initial_w, initial_m, initial_v, beta1, beta2, batch_size, num_batches, max_iters, gamma):
    """
    Do gradient descent using logistic regression. Return the final loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    w = initial_w.reshape(-1,1)
    m = initial_m.reshape(-1,1)
    v = initial_v.reshape(-1,1)
    losses = []
    losses_val = []
    threshold = 1e-4

    y = y.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    eps = 1e-15  # to prevent log(0)

    grads = []
    losses_batch=[]
    losses_val_batch=[]
    for iter in range(max_iters): 
        for batch_y, batch_tx in batch_iter(y, tx, batch_size = batch_size, num_batches=num_batches, shuffle=True):            #batch_size = distance between batch elements ; num_batches = number of elements in the batch
            sig = sigmoid(batch_tx @ w)
            sig = np.clip(sig, 1e-15, 1 - 1e-15)
            N = batch_y.shape[0]
            loss = -(1/N) * (batch_y.T @ np.log(sig) + (1 - batch_y).T @ np.log(1 - sig))
            loss = np.squeeze(loss)
            losses_batch.append(loss)
            grad = (1/N) * batch_tx.T @ (sig - batch_y)
            grads.append(grad)
        stoch_grad = np.mean (grads, axis =0)
        losses.append(np.mean(losses_batch))

        # Validation
        for batch_y, batch_tx in batch_iter( y_val, x_val, batch_size,num_batches=num_batches, shuffle=True):  
            sig_val = sigmoid(batch_tx @ w)
            sig_val = np.clip(sig_val, 1e-15, 1 - 1e-15)
            N_val = batch_y.shape[0]
            loss_val = -(1/N_val) * (batch_y.T @ np.log(sig_val) + (1 - batch_y).T @ np.log(1 - sig_val))
            loss_val = np.squeeze(loss_val)
            losses_val_batch.append(loss_val)
        losses_val.append(np.mean(losses_val_batch))
        t = iter + 1
        m = beta1*m + (1-beta1)*stoch_grad
        v = beta2*v + (1-beta2)*stoch_grad**2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        w = w - gamma * m_hat / (np.sqrt(v_hat) + eps)
    return w, losses, losses_val

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    d = degree+1
    xc=x.reshape(-1,1)
    phi = np.zeros((xc.shape[0], d))
    for j in range(d):   
        phi[:,j]=xc[:,0]**j
    return phi
