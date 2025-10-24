import numpy as np
from common import batch_iter, sigmoid

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        e = y - np.dot(tx,w)
        g = - 1/len(y) * (np.dot(tx.T,e))
        N = y.shape[0]
        loss = 1/(2*N) * np.sum((y-np.dot(tx,w))**2)
        # update w 
        w=w-gamma*g

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w
    for n_iter in range(max_iters):
        # implement stochastic gradient descent.
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):            #batch_size = distance between batch elements ; num_batches = number of elements in the batch
            loss= (batch_y-np.dot(batch_tx,w))**2
            e = batch_y - np.dot(batch_tx,w)
            stoch_grad = - 2 * (np.dot(batch_tx.T,e))

        w=w-gamma*stoch_grad
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # least squares
    A=tx.T@tx
    b= tx.T@y
    w=np.linalg.solve(A,b)

    e = y-tx@w
    N = len(e)
    mse = 1/(2*N)*e.T@e
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = tx.shape[0]
    D = tx.shape[1]
    
    A=tx.T@tx + 2*lambda_*N*np.identity(D)
    b= tx.T@y
    w=np.linalg.solve(A,b)

    e = y-tx@w
    mse = 1/(2*N)*e.T@e
    return w, mse

def logistic_regression(y, tx, y_val, x_val, initial_w, max_iters, gamma):
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
    w=initial_w
    losses=[]
    losses_val=[]
    threshold=1e-8

    y = y.reshape(-1, 1)  # from (N,) → (N,1)

    for iter in range(max_iters):
        sig = sigmoid(tx@w)
        N = y.shape[0]
        loss = -(1/N) * (y.T @ np.log(sig) + (1-y).T @ np.log(1-sig))
        loss = np.squeeze(loss)             # from shape (1,1) → scalar
        losses.append(loss)

        sig_val = sigmoid(x_val@w)
        N=y_val.shape[0]
        loss_val = -(1/N) * (y_val.T @ np.log(sig_val) + (1-y_val).T @ np.log(1-sig_val))
        loss_val = np.squeeze(loss_val)             # from shape (1,1) → scalar
        losses_val.append(loss_val)

        if len(losses_val) > 1 and np.abs(losses_val[-1] - losses_val[-2]) < threshold:
            break

        grad = (1/N) * tx.T@(sig-y) 
        w = w - gamma * grad

    return w, losses, losses_val

def reg_logistic_regression(y_tr, x_tr, y_val, x_val, lambda_, initial_w, max_iters, gamma):
    """Do gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar
        max_iters: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    N = y_tr.shape[0]
    w=initial_w
    threshold = 1e-8
    tr_losses = []
    val_losses = []
 
    y_tr = y_tr.reshape(-1, 1)  # from (N,) → (N,1)
    y_val = y_val.reshape(-1, 1)

    # calcolo pesi di classe basati sul training set
    classes, counts = np.unique(y_tr, return_counts=True)
    K = len(classes)
    class_weights = {c: N / (K * n) for c, n in zip(classes, counts)}

    # array dei pesi per ogni campione
    sample_weights_tr = np.array([class_weights[y_i[0]] for y_i in y_tr]).reshape(-1,1)
    sample_weights_val = np.array([class_weights[y_i[0]] for y_i in y_val]).reshape(-1,1)

    for iter in range(max_iters):
        sig = sigmoid(x_tr @ w)
        #loss = -(1/N) * (y_tr.T @ np.log(sig) + (1-y_tr).T @ np.log(1-sig))
                # cross-entropy pesata + L2
        loss = -(1/N) * np.sum(sample_weights_tr * (y_tr * np.log(sig + 1e-9) + 
                                                        (1 - y_tr) * np.log(1 - sig + 1e-9))) \
                  + lambda_ * np.sum(w**2)
        loss=np.squeeze(loss)
        tr_losses.append(loss)

        #if len(tr_losses) > 1 and np.abs(tr_losses[-1] - tr_losses[-2]) < threshold:
        #    break

        sig_val = sigmoid(x_val @ w)
        N_val = y_val.shape[0]
        #loss_val = -(1 / N_val) * (y_val.T @ np.log(sig_val) + (1 - y_val).T @ np.log(1 - sig_val))
        loss_val = -(1/N_val) * np.sum(sample_weights_val * (y_val * np.log(sig_val + 1e-9) + 
                                                            (1 - y_val) * np.log(1 - sig_val + 1e-9))) \
                   + lambda_ * np.sum(w**2)
        loss_val = np.squeeze(loss_val)
        val_losses.append(loss_val)

        if len(val_losses) > 1 and np.abs(val_losses[-1] - val_losses[-2]) < threshold:
            break

        #grad = (1/N) * x_tr.T@(sig-y_tr) + 2 * lambda_ * w
        # Gradiente pesato + regolarizzazione
        grad = (1/N) * (x_tr.T @ (sample_weights_tr * (sig - y_tr))) + 2 * lambda_ * w

        w = w - gamma * grad
        
    return w, tr_losses, val_losses