from helpers import load_csv_data, create_csv_submission
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from common import batch_iter, sigmoid
from mie import build_k_indices, cross_validation, logistic_loss, train_val_split, subsample_class
from performance import accuracy, precision, recall, f1_score


### LOAD DATA

data_path='data/dataset/dataset'
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)
print("Number of samples of train: ", x_train.shape[0])
print("Number of features: ", x_train.shape[1])
print("Number of samples of test: ", x_test.shape[0])
print("Data type x_train:", x_train.dtype) #float64
print("Data type y_train:", y_train.dtype) #int64

### VARIABLES

threshold = 20      # Fixed threshold
print(f"Threshold fixed at {threshold} unique values\n")
categorical_idx = []
continuous_idx = []
unique_counts = []

# Compute unique values for every column
for i in range(x_train.shape[1]):
    col = x_train[:, i]
    count = len(np.unique(col))
    unique_counts.append(count)

    if count <= threshold:
        categorical_idx.append(i)
    else:
        continuous_idx.append(i)

# Classifies features as categorical or continuous
print(f"Total categorical features (<= {threshold} unique): {len(categorical_idx)}")
print(f"Total continuous features    (>  {threshold} unique): {len(continuous_idx)}")

# Build histrogram
unique_counts = np.array(unique_counts)
unique_vals, counts = np.unique(unique_counts, return_counts=True)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(unique_vals, counts, color='lightblue', width=1.0)
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Number of unique values')
plt.xlim(right=200)
plt.xlim(left=0)
plt.ylabel('Number of features')
plt.title('Distribution of unique values per feature')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

### CLEANING

missing_counts = np.sum(np.isnan(x_train), axis=0)
missing_percents = missing_counts / x_train.shape[0] * 100

print(f"Number of features with more than 40% of NaN: {np.sum(missing_percents>40)}")

# Drop feature if at least 40% are NaN values
valid_cols_nan = np.where(np.isnan(x_train).sum(axis=0) < 0.4 * x_train.shape[0])[0]
x_train_clean = x_train[:, valid_cols_nan]
x_test_clean = x_test[:, valid_cols_nan]
print(f"Shape after NaN cleaning: {x_train_clean.shape}")

# For other columns, NaN are replaced with the mean
col_mean = np.nanmean(x_train_clean, axis = 0)  # np.nanmean ignores the NaN
inds =np.where(np.isnan(x_train_clean))
x_train_clean[inds] = col_mean[inds[1]]        #x_train_clean[inds] = np.take(col_mean, inds[1])
inds_t =np.where(np.isnan(x_test_clean))
x_test_clean[inds_t] = col_mean[inds_t[1]]


### DECORRELATION

# Cross-correlation among features
def corrcoef(X):
    n_features = X.shape[1]
    corr = np.empty((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            corr_ij = np.corrcoef(X[:, i], X[:, j])[0, 1]       # np.corrcoef returns correlation matrix of two 1D-arrays, corr_ij is in position [0,1]            
            corr[i, j] = corr_ij
            corr[j, i] = corr_ij  
    return corr

corr_matrix = corrcoef(x_train_clean)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Correlation matrix before cleaning')
plt.show()

# Drop features if correlation > 0.9

non_valid_col = set()  # Use set to avoid duplicates
for i in range(x_train_clean.shape[1]):
    for j in range(i + 1, x_train_clean.shape[1]):
        if abs(corr_matrix[i, j]) > 0.9:
            non_valid_col.add(j)  # Drop j, keep i

x_train_decorr = np.delete(x_train_clean, list(non_valid_col), axis=1)     # Remove column
x_test_decorr = np.delete(x_test_clean, list(non_valid_col), axis=1)
print(f"Shape of decorrelated X: {x_train_decorr.shape}")


### LOW INFO

def drop_low_info_features(X, cont_indices, cat_indices, var_thresh, mode_thresh):
    """
    Removes low-variance features (for continuous variables)
    and high-mode features (for categorical variables),

    """
    
    n_samples = X.shape [0]
    n_features = X.shape [1]
    keep_mask = np.ones(n_features, dtype=bool)
    
    # 1️  continuos feature: variance
    for i in cont_indices:
        var = np.var(X[:, i].astype(float))
        if var < var_thresh:
            keep_mask[i] = False

    # 2️  categorical feature:  mode
    for i in cat_indices:
        col = X[:, i]
        # Find the most frequent value (mode) and its relative frequency, if the latter overcomes a threshold the feaature is dropped since non-informative
        values, counts = np.unique(col, return_counts=True)
        freq_max = counts.max() / n_samples
        if freq_max > mode_thresh:
            keep_mask[i] = False

    # 3️  Filtering valid columns
    X_filtered = X[:, keep_mask]
    
    print(f"Shape after feature removal: {X_filtered.shape}")
    return X_filtered

# Fixed threshold
threshold = 20
categorical_idx = []
continuous_idx = []
unique_counts = []

# Compute unique values for every column
for i in range(x_train_decorr.shape[1]):
    col = x_train_decorr[:, i]
    count = len(np.unique(col))
    unique_counts.append(count)

    if count <= threshold:
        categorical_idx.append(i)
    else:
        continuous_idx.append(i)

x_train_inf= drop_low_info_features(x_train_decorr, cont_indices=continuous_idx, cat_indices=categorical_idx, var_thresh=1e-3, mode_thresh=0.95)
x_test_inf= drop_low_info_features(x_test_decorr, cont_indices=continuous_idx, cat_indices=categorical_idx, var_thresh=1e-3, mode_thresh=0.95)


### DECORRELATION WITH OUTPUT 
correlations = np.empty(x_train_inf.shape[1])

for i in range(x_train_inf.shape[1]):
    correlations[i] = np.corrcoef(x_train_inf[:, i], y_train)[0, 1]
    
valid_features=np.where(np.abs(correlations)>=0.05)[0]
x_train_denoised=x_train_inf[:,valid_features]
x_test_denoised=x_test_inf[:,valid_features]

print(f"Shape after denoising: {x_train_denoised.shape}")

### LINEAR DEPENDENCE
rank = np.linalg.matrix_rank(x_train_denoised)

print(f"Number of features: {x_train_denoised.shape[1]}")
print(f"Rank of the matrix: {rank}")

if rank < x_train_denoised.shape[1]:
    print("Some features are linearly dependent!")


### Z-SCORE
col_mean_0=np.mean(x_train_denoised, axis=0)
col_sd_0=np.std(x_train_denoised, axis=0)

x_train_norm = (x_train_denoised-col_mean_0) / col_sd_0
x_test_norm = (x_test_denoised - col_mean_0) / col_sd_0

col_mean=np.mean(x_train_norm, axis=0)
col_sd=np.std(x_train_norm, axis=0)

print(f"Before normalization, mean: {col_mean_0[0:5]}, sd: {col_sd_0[0:5]}")
print(f"After normalization, mean: {col_mean[0:5]}, sd: {col_sd[0:5]}")


### CLASS DISTRIBUTION
plt.hist(y_train, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribuzione delle etichette (y_train)")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.show()

### SPLIT AND SUBSAMPLING

# Split
X_train, y_train, X_val, y_val = train_val_split(x_train_norm, y_train, val_ratio=0.2)

# Subsampling
X_bal, y_bal = subsample_class(X_train, y_train, target_ratio=1.0)


### CROSS-VALIDATION
k_fold=5
k_indices=build_k_indices(y_bal, k_fold, seed=42)
loss_val=[]
loss_tr=[]
ws=[]
initial_w = np.zeros((X_bal.shape[1], 1))

for k in range(k_fold):
    w, loss_tr_tmp=cross_validation(y_bal, X_bal, k_indices, k, initial_w, max_iters=1000, gamma=0.1, lambda_=1e-4)
    loss_tr.append(loss_tr_tmp)
    ws.append(w)
    loss_val_tmp=logistic_loss(y_val, X_val, w)
    loss_val.append(loss_val_tmp)
    
w_best=np.mean(ws, axis=0)


### INFERENCE

y_pred_prob = sigmoid(x_test_norm @ w_best)
y_pred = np.where(y_pred_prob >= 0.5, 1, -1)

create_csv_submission(test_ids, y_pred, 'Reg_Logistic_1')

### METRICS ON VALIDATION

y_val_pred_prob = sigmoid(X_val @ w_best)
y_val_pred = np.where(y_val_pred_prob >= 0.5, 1, -1)

acc = accuracy(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1-score: {f1:.4f}")
