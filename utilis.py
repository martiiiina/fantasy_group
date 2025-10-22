import numpy as np
from implementations import logistic_regression

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

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

def sigmoid(t):
    sigma = 1 / (1+np.exp(-t))
    return sigma

def logistic_loss(y,tx,w):
    N = y.shape[0]
    loss = -(1/N) * np.sum(y * np.log(sigmoid(tx@w)) + (1-y) * np.log(1-sigmoid(tx@w)))
    loss=loss.item()
    return loss

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

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]      # fa lista da 0 a K_fold, == k corrente quindi lista di zeri e un uno per k corrente, invertiti con la tilde
    
    # N.B. passando un booleano a k_indices del tipo T T F T T, si fa slicing delle righe di k_indices con indice True

    tr_indice = tr_indice.reshape(-1)   # trasforma tr_indice da matrice e array 1D

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]

    w, loss_tr=logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)

    return w, loss_tr