import numpy as np
from implementations import logistic_regression, reg_logistic_regression
from common import sigmoid, batch_iter

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

def cross_validation(y, x, k_indices, k, initial_w, max_iters, gamma, lambda_):
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
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]      # list from 0 to K_fold, == current k, list of zeros and one 1 for current k, inverted with tilde
    
    # N.B. passing a boolean to k_indices like T T F T T, slicing of rows of k_indices with index True

    tr_indice = tr_indice.reshape(-1)   # trasforms tr_indice from matrix to 1D

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice, :]
    x_tr = x[tr_indice, :]
    #print("Shape after split x_tr: ", x_tr.shape)
    #print("Shape after split y_tr: ", y_tr.shape)


    w, loss_tr=reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

    return w, loss_tr