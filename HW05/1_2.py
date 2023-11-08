import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def k_means_clustering(data, k, max_iterations=100):
    # Initialize the centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for iteration in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(
            data[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update the centroids by taking the mean of the assigned data points
        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


def evl_obj(data, centroids, labels):
    # Compute the objective of the clustering
    obj = 0
    for i in range(len(data)):
        obj += np.linalg.norm(data[i] - centroids[labels[i]]) ** 2

    return obj


def evl_acc(means, centroids, labels, true_labels):
    # Compute the accuracy of the clustering
    # First map each centroid to a cluster based on the true means
    mapping = {}
    for i in range(len(centroids)):
        min_dist = np.inf
        for j in range(len(means)):
            dist = np.linalg.norm(centroids[i] - means[j])
            if dist < min_dist:
                min_dist = dist
                mapping[i] = j
    # Convert all the labels based on this mapping
    for i in range(len(labels)):
        labels[i] = mapping[labels[i]]

    # Compute the accuracy
    acc = 0
    for i in range(len(labels)):
        if labels[i] == true_labels[i]:
            acc += 1

    return acc / len(labels)


# =====================================
# GMM
# =====================================


def initialize_parameters(data, n_components):
    n_samples, n_features = data.shape
    weights = np.full(n_components, 1.0 / n_components)
    means = data[np.random.choice(n_samples, n_components, replace=False)]
    covariances = [np.cov(data, rowvar=False)] * n_components
    return weights, means, covariances


def e_step(data, weights, means, covariances, n_components):
    n_samples, _ = data.shape
    responsibilities = np.zeros((n_samples, n_components))
    for k in range(n_components):
        multivariate = multivariate_normal(
            means[k], covariances[k], allow_singular=True)
        responsibilities[:, k] = weights[k] * multivariate.pdf(data)
    responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
    return responsibilities


def m_step(data, responsibilities, n_components):
    n_samples, n_features = data.shape
    weights = responsibilities.sum(axis=0) / n_samples
    means = np.dot(responsibilities.T, data) / \
        responsibilities.sum(axis=0)[:, np.newaxis]
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = data - means[k]
        covariances[k] = np.dot(responsibilities[:, k]
                                * diff.T, diff) / responsibilities[:, k].sum()
    return weights, means, covariances


def gmm(data, n_components, max_iterations=100, tol=1e-4):
    weights, means, covariances = initialize_parameters(data, n_components)

    for iteration in range(max_iterations):
        old_means = means.copy()

        # E-step
        responsibilities = e_step(
            data, weights, means, covariances, n_components)

        # M-step
        weights, means, covariances = m_step(
            data, responsibilities, n_components)

        # Check for convergence
        mean_diff = np.linalg.norm(means - old_means)
        if mean_diff < tol:
            break

    return weights, means, covariances

# Assign the labels based on


def assign_labels(data, weights, means, covariances):
    n_components = len(weights)
    neg_log_likelihoods = np.zeros((len(data), n_components))

    for k in range(n_components):
        multivariate = multivariate_normal(
            means[k], covariances[k], allow_singular=True)
        neg_log_likelihoods[:, k] = -1 * \
            (np.log(weights[k]) + multivariate.logpdf(data))

    labels = np.argmin(neg_log_likelihoods, axis=1)
    return labels


def evl_gmm_obj(data, weights, means, covariances):
    n_samples, _ = data.shape
    n_components = len(weights)

    neg_log_likelihood = 0.0
    for i in range(n_samples):
        log_likelihood = 0.0
        for k in range(n_components):
            multivariate = multivariate_normal(
                means[k], covariances[k], allow_singular=True)
            log_likelihood += weights[k] * multivariate.pdf(data[i])
        neg_log_likelihood += -np.log(log_likelihood)

    return neg_log_likelihood


if __name__ == "__main__":
    # Sample 100 points from a normal distribution

    sigmas = [0.5, 1, 2, 4, 8]

    k_means_clustering_objs = []
    k_means_accuracies = []
    gmm_clustering_objs = []
    gmm_accuracies = []

    for sigma in sigmas:
        cov_1 = np.array([[2, 0.5], [0.5, 1]])
        cov_1 = sigma * cov_1
        mean_1 = np.array([-1, -1])
        p_a = np.random.multivariate_normal(mean=[-1, -1], cov=cov_1, size=100)

        cov_2 = np.array([[1, -0.5], [-0.5, 2]])
        cov_2 = sigma * cov_2
        mean_2 = np.array([1, -1])
        p_b = np.random.multivariate_normal(mean=[1, -1], cov=cov_2, size=100)

        cov_3 = np.array([[1, 0], [0, 2]])
        cov_3 = sigma * cov_3
        mean_3 = np.array([0, 1])
        p_c = np.random.multivariate_normal(mean=[0, 1], cov=cov_3, size=100)

        # Combine all three distributions in one array
        p = np.concatenate((p_a, p_b, p_c), axis=0)
        true_means = np.array([mean_1, mean_2, mean_3])
        true_labels = np.concatenate(
            (np.zeros(100), np.ones(100), np.ones(100) * 2))

        # =====================================
        # K-means clustering
        # =====================================
        centroids, labels = k_means_clustering(p, 3, max_iterations=100)

        # Evaluate the clustering objective
        k_means_clustering_obj = evl_obj(p, centroids, labels)

        # Calculate the accuracy of the clustering
        k_means_accuracy = evl_acc(true_means, centroids, labels, true_labels)

        k_means_clustering_objs.append(k_means_clustering_obj)
        k_means_accuracies.append(k_means_accuracy)
        print("K-means clustering objective: {}".format(k_means_clustering_obj))
        print("K-means clustering accuracy: {}".format(k_means_accuracy))

        # =====================================
        # GMM
        # =====================================
        weights, means, covarainces = gmm(
            p, 3, max_iterations=100, tol=1e-4)

        # Assign labels to each data point based on the GMM
        labels = assign_labels(p, weights, means, covarainces)

        # Evaluate the clustering objective
        gmm_clustering_obj = evl_gmm_obj(p, weights, means, covarainces)

        # Calculate the accuracy of the clustering
        gmm_accuracy = evl_acc(true_means, means, labels, true_labels)

        gmm_clustering_objs.append(gmm_clustering_obj)
        gmm_accuracies.append(gmm_accuracy)
        print("GMM clustering objective: {}".format(gmm_clustering_obj))
        print("GMM clustering accuracy: {}".format(gmm_accuracy))


# Plot the results
plt.figure()
plt.plot(sigmas, k_means_clustering_objs, label="K-means")
plt.plot(sigmas, gmm_clustering_objs, label="GMM")
plt.xlabel("Sigma")
plt.ylabel("Clustering objective")
plt.legend()
plt.savefig("./images/1_2_clustering_obj.png")
plt.show()

plt.figure()
plt.plot(sigmas, k_means_accuracies, label="K-means")
plt.plot(sigmas, gmm_accuracies, label="GMM")
plt.xlabel("Sigma")
plt.ylabel("Clustering accuracy")
plt.legend()
plt.savefig("./images/1_2_clustering_acc.png")
plt.show()
