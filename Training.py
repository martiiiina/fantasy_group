def main():
    # Import libraries
    from helpers import create_csv_submission
    import numpy as np
    import matplotlib.pyplot as plt
    from support import build_k_indices, cross_validation, subsample_class, sigmoid

    # Load data
    x_train = np.load("processed/x_train.npy")
    x_test = np.load("processed/x_test.npy")
    y_train = np.load("processed/y_train.npy")

    # Mapping labels from -1/1 to 0/1
    y_train = (y_train + 1) / 2 

    # Add bias term to X (column of 1)
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])

    # Undersampling to balance classes
    x_train, y_train = subsample_class(x_train, y_train, target_ratio=1.0)

    # Cross validation
    k_fold=5
    k_indices=build_k_indices(y_train, k_fold, seed=42)

    loss_val=[]
    ws=[]
    initial_w = np.zeros((x_train.shape[1], 1))
    
    # Hyperparameters
    max_iters=3000
    gamma=0.1
    lambda_=0

    for k in range(k_fold):
        w, loss_tr_tmp, loss_val_tmp = cross_validation(y_train,x_train,k_indices,k,initial_w,max_iters,gamma,lambda_)
        best_val_loss = np.min(loss_val_tmp)
        loss_val.append(best_val_loss)
        ws.append(w)

        plt.plot(loss_tr_tmp, label='Training Loss')
        plt.plot(loss_val_tmp, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss per Epoch for fold {k+1}')
        plt.legend()
        plt.show()
    w_best=np.mean(ws, axis=0)
    gen_err = np.mean(loss_val)
    print("Generalization error: ", gen_err)

    # Save weights
    np.save("w_best.npy", w_best)

    

if __name__ == "__main__":
    main()
