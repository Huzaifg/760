import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# plt.rcParams.update({
#     'axes.titlesize': 16,
#     'axes.labelsize': 14,
#     'lines.linewidth': 1.5,
#     'lines.markersize': 6,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'Palatino', 'serif'],
#     # "font.serif" : ["Computer Modern Serif"],
# })

# =====================================
# Buggy-PCA
# =====================================


def buggy_pca(X, d):
    # Step 1: Compute the SVD of X
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    Z = np.dot(X, VT.T[:, :d])
    reconstruction = np.dot(Z, VT[:d, :])

    return Z, VT, reconstruction

# =====================================
# Demeaned-PCA
# =====================================


def demeaned_pca(X, d):
    # Step 1: Subtract the mean along each dimension
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean

    Z, VT, reconstruction = buggy_pca(X_demeaned, d)
    reconstruction = reconstruction + mean

    return Z, VT, reconstruction

# =====================================
# Normalized-PCA
# =====================================


def normalized_pca(X, d):
    # Step 1: Subtract the mean along each dimension
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean

    # Step 2: Scale each dimension to have sample mean of 0 and standard deviation of 1
    std_dev = np.std(X, axis=0)
    X_normalized = X_demeaned / std_dev

    Z, VT, reconstruction = buggy_pca(X_normalized, d)
    reconstruction = np.dot(Z, VT[:d, :]) * std_dev + mean

    return Z, VT, reconstruction


# =====================================
# DRO
# =====================================
def DRO(data, d):
    n, D = data.shape

    # Step 1: Find the value for b in the optimal solution
    b = np.mean(data, axis=0)

    # Step 2: Define Y = X - b
    Y = data - b

    # Step 3: Take the Singular Value Decomposition of Y
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)

    # Get the diagonal elements of S and also return those
    S_diag = S

    # Truncate the SVD components to d dimensions
    U = U[:, :d]
    S = np.diag(S[:d])
    Vt = Vt[:d, :]

    # Step 4: Map the truncated SVD components back to the original problem
    At = np.dot(S, Vt)
    Z = U

    # Calculate the d-dimensional representations
    d_dim_representations = Z  # Each column of Z is a d-dimensional representation

    # Calculate the reconstructions in D dimensions
    reconstructions = Z @ At + b

    return d_dim_representations, At, b, reconstructions, S_diag


if __name__ == "__main__":

    # Load data from csv file into numpy arrays
    d1 = np.loadtxt('data/data2D.csv', delimiter=',')
    d2 = np.loadtxt('data/data1000D.csv', delimiter=',')

    # Call all methods on d1
    buggy_representation, buggy_parameters, buggy_reconstruction = buggy_pca(
        d1, 1)
    demeaned_representation, demeaned_parameters, demeaned_reconstruction = demeaned_pca(
        d1, 1)
    normalized_representation, normalized_parameters, normalized_reconstruction = normalized_pca(
        d1, 1)
    dro_representation, dro_parameters_A, dro_parameters_b, dro_reconstruction, S_diag = DRO(
        d1, 1)

    # Plot the results
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.title('Buggy PCA')
    plt.scatter(d1[:, 0], d1[:, 1], alpha=0.5)
    plt.scatter(buggy_reconstruction[:, 0],
                buggy_reconstruction[:, 1], alpha=0.5)
    plt.legend(['Data', 'Reconstruction'])
    plt.subplot(1, 4, 2)
    plt.title('Demeaned PCA')
    plt.scatter(d1[:, 0], d1[:, 1], alpha=0.5)
    plt.scatter(demeaned_reconstruction[:, 0],
                demeaned_reconstruction[:, 1], alpha=0.5)
    plt.legend(['Data', 'Reconstruction'])
    plt.subplot(1, 4, 3)
    plt.title('Normalized PCA')
    plt.scatter(d1[:, 0], d1[:, 1], alpha=0.5)
    plt.scatter(normalized_reconstruction[:, 0],
                normalized_reconstruction[:, 1], alpha=0.5)
    plt.legend(['Data', 'Reconstruction'])
    plt.subplot(1, 4, 4)
    plt.title('DRO')
    plt.scatter(d1[:, 0], d1[:, 1], alpha=0.5)
    plt.scatter(dro_reconstruction[:, 0],
                dro_reconstruction[:, 1], alpha=0.5)
    plt.legend(['Data', 'Reconstruction'])
    plt.savefig('./images/2_3.png')
    fig.tight_layout()
    plt.show()

    # Now use d2 and first get the knee point from DRO
    _, _, _, _, S_ = DRO(d2, 1)

    # Plot the singular values
    plt.figure()
    plt.plot(S_)
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.savefig('./images/2_3_singular_values.png')
    plt.show()

    # Print index where values in S_ are less than 30
    d = np.where(S_ < 30)[0][0]
    print("Knee point: {}".format(d))

    # Compute the reconstruction error for each method which is the forbenius norm of the difference between the original data and the reconstruction

    buggy_reconstruction_error = np.sum((d1 - buggy_reconstruction)**2)

    demeaned_reconstruction_error = np.sum((d1 - demeaned_reconstruction)**2)
    normalized_reconstruction_error = np.sum(
        (d1 - normalized_reconstruction)**2)
    dro_reconstruction_error = np.sum((d1 - dro_reconstruction)**2)

    print("Buggy reconstruction error dataset1: {}".format(
        buggy_reconstruction_error))
    print("Demeaned reconstruction error dataset1: {}".format(
        demeaned_reconstruction_error))
    print("Normalized reconstruction error dataset1: {}".format(
        normalized_reconstruction_error))
    print("DRO reconstruction error dataset1: {}".format(dro_reconstruction_error))

    # Call all methods on d2
    buggy_representation, buggy_parameters, buggy_reconstruction = buggy_pca(
        d2, d+1)
    demeaned_representation, demeaned_parameters, demeaned_reconstruction = demeaned_pca(
        d2, d+1)
    normalized_representation, normalized_parameters, normalized_reconstruction = normalized_pca(
        d2, d+1)
    dro_representation, dro_parameters_A, dro_parameters_Z, dro_reconstruction, S_diag = DRO(
        d2, d+1)

    # Compute the errors
    buggy_reconstruction_error = np.sum((d2 - buggy_reconstruction)**2)
    demeaned_reconstruction_error = np.sum((d2 - demeaned_reconstruction)**2)
    normalized_reconstruction_error = np.sum(
        (d2 - normalized_reconstruction)**2)
    dro_reconstruction_error = np.sum((d2 - dro_reconstruction)**2)

    print("Buggy reconstruction error dataset1: {}".format(
        buggy_reconstruction_error))
    print("Demeaned reconstruction error dataset1: {}".format(
        demeaned_reconstruction_error))
    print("Normalized reconstruction error dataset1: {}".format(
        normalized_reconstruction_error))
    print("DRO reconstruction error dataset1: {}".format(dro_reconstruction_error))
