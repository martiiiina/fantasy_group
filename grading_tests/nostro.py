import pathlib
from implementations_submit import mean_squared_error_gd
import numpy as np
import pytest
from conftest import ATOL, GITHUB_LINK, RTOL
FUNCTIONS = [
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
]
MAX_ITERS = 2
GAMMA = 0.1

@pytest.fixture()
def initial_w():
    return np.array([0.5, 1.0])


@pytest.fixture()
def y():
    return np.array([0.1, 0.3, 0.5])


@pytest.fixture()
def tx():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])



def test_mean_squared_error_gd_0_step(y, tx):
    expected_w = np.array([0.413044, 0.875757])
    #w, loss = student_implementations.mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)
    w, loss = mean_squared_error_gd(y,tx,expected_w,0,GAMMA)
    expected_w = np.array([0.413044, 0.875757])
    expected_loss = 2.959836

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape