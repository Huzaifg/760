import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# Step 1: Generate the training set
np.random.seed(0)  # Set a random seed for reproducibility
a, b = 0, 2 * np.pi  # Interval [0, 2*pi]
n = 100  # Number of training points
x_train = np.sort(np.random.uniform(a, b, n))
y_train = np.sin(x_train)

# Step 2: Build a model f using Lagrange interpolation
lagrange_model = interp.lagrange(x_train, y_train)

# Step 3: Generate a test set with the same distribution
x_test = np.sort(np.random.uniform(a, b, n))
y_test = np.sin(x_test)

# Step 4: Compute and report training and test errors
y_train_predicted = lagrange_model(x_train)
y_test_predicted = lagrange_model(x_test)

train_error = np.log(np.mean(np.square(y_train - y_train_predicted)))
test_error = np.log(np.mean(np.square(y_test - y_test_predicted)))

print("Training Error:", train_error)
print("Test Error:", test_error)

# Step 5: Repeat the experiment with noise
# Different noise standard deviations
noise_stddev_values = [0.01, 0.1, 0.5, 1.0]

for noise_stddev in noise_stddev_values:
    # Add zero-mean Gaussian noise to x_test
    noisy_x_test = x_test + np.random.normal(0, noise_stddev, n)
    y_noisy_test = np.sin(noisy_x_test)

    # Compute and report test error with noise
    y_noisy_test_predicted = lagrange_model(noisy_x_test)
    noisy_test_error = np.log(np.mean(
        np.square(y_noisy_test - y_noisy_test_predicted)))

    print(
        f"Noise Stddev: {noise_stddev}, Test Error with Noise: {noisy_test_error}")

# Plot the original function and interpolated function
x_plot = np.linspace(a, b, 1000)
y_true = np.sin(x_plot)
y_interpolated = lagrange_model(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, label="True Function (sin(x))", linewidth=2)
plt.plot(x_plot, y_interpolated, label="Interpolated Function",
         linestyle='--', linewidth=2)
plt.scatter(x_train, y_train, label="Training Points", c='r', s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Lagrange Interpolation")
plt.grid(True)
plt.show()
