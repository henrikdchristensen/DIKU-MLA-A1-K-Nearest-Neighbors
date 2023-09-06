import numpy as np
import matplotlib.pyplot as plt
from func.knn import knn


def validation(data: np.ndarray, labels: np.array, m: int, n_values: list, num_validation_sets: int, output_path: str):
    # Define training set and labels
    training_points = data[:m]
    training_labels = labels[:m]

    K = np.arange(1, m + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    for n in n_values:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        mean_validation_errors_of_sets = []

        for i in range(num_validation_sets):
            # Define validation set
            start_idx = m + i * n
            end_idx = m + (i + 1) * n

            validation_points = data[start_idx:end_idx]
            validation_labels = labels[start_idx:end_idx]

            # Calculate validation error
            validation_errors_of_set = np.zeros(m)

            for validation_point, validation_label in zip(validation_points, validation_labels):
                validation_error = knn(training_points, training_labels, validation_point, validation_label)
                validation_errors_of_set += validation_error

            # Divide by n to get the average error for each K
            mean_validation_errors_of_set = validation_errors_of_set / n

            ax2.plot(K, mean_validation_errors_of_set, label=f"Validation set {i+1}")

            # Calculate variance of validation error for comparing validation sets
            mean_validation_errors_of_sets.append(mean_validation_errors_of_set)

        # Plot the mean validation error as a function of K
        ax2.set_xlabel("K")
        ax2.set_ylabel("Validation error")
        ax2.set_title(f"Validation error for K=(1,...,{m}), m={m}, n={n}")
        ax2.grid(alpha=0.2)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(output_path + f"/validation_err_n={n}.png")

        # Stack the validation errors for each set, so that we can calculate the variance
        mean_validation_errors_stacked_of_sets = np.vstack(mean_validation_errors_of_sets)
        variance_of_sets = np.var(mean_validation_errors_stacked_of_sets, axis=0)

        # Plot the variance of the validation error as a function of K
        ax1.plot(K, variance_of_sets, label=f"n = {n}")
        ax1.set_xlabel("K")
        ax1.set_ylabel("Variance of validation error")
        ax1.set_title(f"Variance of validation error for K=(1,...,{m}), m={m}")
        ax1.grid(alpha=0.2)
        ax1.legend()

    fig1.tight_layout()
    fig1.savefig(output_path + f"/variance_of_validation_error.png")

    plt.show()
