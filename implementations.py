import numpy as np


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
        # Loss
        N = len(y)
        loss = 1 / (2 * N) * np.sum((y - np.dot(tx, w)) ** 2)
        # Gradient
        e = y - np.dot(tx, w)
        g = -1 / N * (np.dot(tx.T, e))
        # Update w
        w = w - gamma * g

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
        for batch_y, batch_tx in batch_iter(
            y, tx, batch_size, num_batches=1, shuffle=True
        ):  # batch_size = distance between batch elements ; num_batches = number of elements in the batch
            e = batch_y - np.dot(batch_tx, w)
            stoch_grad = -np.dot(batch_tx.T, e)

        loss = (y - np.dot(tx, w)) ** 2
        w = w - gamma * stoch_grad

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
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)

    e = y - tx @ w
    N = len(e)
    mse = 1 / (2 * N) * e.T @ e

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

    A = tx.T @ tx + 2 * lambda_ * N * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)

    e = y - tx @ w
    mse = 1 / (2 * N) * e.T @ e
    return w, mse


def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    N = y.shape[0]

    for iter in range(max_iters):
        # Predictions
        sig = sigmoid(tx @ w)
        loss = -(1 / N) * (y.T @ np.log(sig) + (1 - y).T @ np.log(1 - sig))
        loss = np.squeeze(loss)
        # Gradient update
        grad = (1 / N) * tx.T @ (sig - y)
        w = w - gamma * grad
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
    N = y.shape[0]
    w = initial_w

    for iter in range(max_iters):
        sig = sigmoid(tx @ w)
        loss = -(1 / N) * (y.T @ np.log(sig) + (1 - y).T @ np.log(1 - sig))
        loss = np.squeeze(loss)
        grad = (1 / N) * tx.T @ (sig - y) + 2 * lambda_ * w
        w = w - gamma * grad

    return w, loss
