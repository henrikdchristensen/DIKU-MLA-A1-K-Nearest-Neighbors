from func.do_validation import do_validation
import numpy as np

def main():
    # Load labels globally - same will be used for all the experiments
    labels_file_path = "data/MNIST-5-6-Subset-Labels.txt"
    labels = np.loadtxt(labels_file_path)

    # Convert labels from {5, 6} to {-1, 1}
    labels = np.where(labels == 5, -1, 1)

    # Same number of validation sets for all experiments
    num_validation_sets = 5

    # Load data
    base_data_file_path = "data/MNIST-5-6-Subset.txt"
    base_output_path = "output/base"
    base_data = np.loadtxt(base_data_file_path).reshape(1877, 784)

    # Same number of validation sets for all experiments
    num_validation_sets = 5

    n_values = [10, 20, 40, 80]  # Size of validation set
    m = 50

    do_validation(base_data, labels, m, n_values, num_validation_sets, base_output_path)

if __name__ == "__main__":
    main()