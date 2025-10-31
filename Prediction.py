def main():
    import numpy as np
    from support import sigmoid
    from helpers import create_csv_submission

    test_ids = np.load("processed/test_ids.npy")
    x_test = np.load("processed/x_test.npy")
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    w_best = np.load("w_best.npy")   
    
    # Inference
    y_pred_prob = sigmoid(x_test @ w_best)
    y_pred = np.where(y_pred_prob >= 0.5, 1, -1)
    create_csv_submission(test_ids, y_pred, 'Train_adam')

if __name__ == "__main__":
    main()
