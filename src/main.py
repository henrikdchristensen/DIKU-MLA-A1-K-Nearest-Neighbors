from knn import KNN


def main():
    uncorrupted = KNN(name="Uncorrupted", data_file_path="data/MNIST-5-6-Subset.txt", output_path="images/uncorrupted", n_values=[10, 20, 40, 80])
    uncorrupted.validation()

    # Light corruption experiment
    light_corruption = KNN(
        name="Light-Corruption",
        data_file_path="data/MNIST-5-6-Subset-Light-Corruption.txt",
        output_path="images/light_corruption",
        n_values=[80],
        skip_variance=True,
    )
    light_corruption.validation()

    # Moderate corruption experiment
    moderate_corruption = KNN(
        name="Moderate-Corruption",
        data_file_path="data/MNIST-5-6-Subset-Moderate-Corruption.txt",
        output_path="images/moderate_corruption",
        n_values=[80],
        skip_variance=True,
    )
    moderate_corruption.validation()

    # Heavy corruption experiment
    heavy_corruption = KNN(
        name="Heavy-Corruption",
        data_file_path="data/MNIST-5-6-Subset-Heavy-Corruption.txt",
        output_path="images/heavy_corruption",
        n_values=[80],
        skip_variance=True,
    )
    heavy_corruption.validation()


if __name__ == "__main__":
    main()
