from func.validation import validation
import numpy as np


def main():
    # Same labels will be used for all the experiments
    labels_file_path = "data/MNIST-5-6-Subset-Labels.txt"
    labels = np.loadtxt(labels_file_path)
    # Convert labels from {5, 6} to {-1, 1}
    labels = np.where(labels == 5, -1, 1)

    # Same number of validation sets for all experiments
    num_validation_sets = 5

    # Same number of training points for all experiments
    m = 50

    # Base data experiment
    base_data_file_path = "data/MNIST-5-6-Subset.txt"
    base_output_path = "output/base"
    base_data = np.loadtxt(base_data_file_path).reshape(1877, 784)
    n_values = [10, 20, 40, 80]  # Size of validation set
    validation(base_data, labels, m, n_values, num_validation_sets, base_output_path)

    # Light corruption experiment
    light_corruption_data_file_path = "data/MNIST-5-6-Subset-Light-Corruption.txt"
    light_corruption_output_path = "output/light_corruption"
    light_corruption_data = np.loadtxt(light_corruption_data_file_path).reshape(1877, 784)
    n_values = [80]
    validation(light_corruption_data, labels, m, n_values, num_validation_sets, light_corruption_output_path)

    # Moderate corruption experiment
    moderate_corruption_data_file_path = "data/MNIST-5-6-Subset-Moderate-Corruption.txt"
    moderate_corruption_output_path = "output/moderate_corruption"
    moderate_corruption_data = np.loadtxt(moderate_corruption_data_file_path).reshape(1877, 784)
    n_values = [80]
    validation(moderate_corruption_data, labels, m, n_values, num_validation_sets, moderate_corruption_output_path)

    # Heavy corruption experiment
    heavy_corruption_data_file_path = "data/MNIST-5-6-Subset-Heavy-Corruption.txt"
    heavy_corruption_output_path = "output/heavy_corruption"
    heavy_corruption_data = np.loadtxt(heavy_corruption_data_file_path).reshape(1877, 784)
    n_values = [80]
    validation(heavy_corruption_data, labels, m, n_values, num_validation_sets, heavy_corruption_output_path)


if __name__ == "__main__":
    main()
