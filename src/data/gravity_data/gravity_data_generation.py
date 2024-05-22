import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import os


def generate_layer_thickness(lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound)


def add_ellipse(model, center, axes, density):
    """
    Adds an ellipse of the given density to the model at the given location.

    Args:
        model (numpy.ndarray): The model to add the ellipse to.
        center (tuple): The center of the ellipse.
        axes (tuple): The length of the two axes of the ellipse.
        density (float): The density of the ellipse.

    Returns:
        numpy.ndarray: The updated model with the ellipse added.
    """
    # Iterate over each grid cell
    for x in range(model.shape[0]):
        for y in range(model.shape[1]):
            # Check if the point (x, y) is inside the ellipse
            if (x - center[0]) ** 2 / axes[0] ** 2 + (y - center[1]) ** 2 / axes[
                1
            ] ** 2 <= 1:
                model[x, y] = density

    return model


def add_fault(model, dip_degrees, throw, fault_x):
    dip = np.radians(dip_degrees)
    height, width = model.shape

    faulted_model = np.copy(model)

    # Apply the fault to the model
    for y in range(height - throw):
        y1 = y
        y2 = y + throw
        offset_1 = int(np.tan(dip) * y1)
        offset_2 = int(np.tan(dip) * y2)
        x_shifted_1 = fault_x + offset_1
        x_shifted_2 = fault_x + offset_2

        if dip_degrees >= 0:
            x_21, x_22 = x_shifted_2, width
            x_11, x_12 = x_shifted_1, width - x_shifted_2 + x_shifted_1

        else:
            x_21, x_22 = 0, x_shifted_2
            x_11, x_12 = x_shifted_1 - x_shifted_2, x_shifted_1

        if 0 <= x_shifted_2 < width:
            if y1 < throw:
                faulted_model[y1, x_11:x_12] = 2.3
            faulted_model[y2, x_21:x_22] = model[y1, x_11:x_12]

    return faulted_model


def generate_models(
    num_models=10000, grid_size=(32, 32), path_save=None, save_model=False
):
    """
    Generates a specified number of gravity models with random layer thicknesses and densities.

    Args:
        num_models (int, optional): The number of models to generate. Defaults to 10000.
        grid_size (tuple, optional): The size of the grid for each model. Defaults to (32, 32).

    Returns:
        list: A list of generated models, where each model is a 2D numpy array representing the density of each grid cell.
    """

    models, gravity_measures, faults = [], [], []
    
    table_bounds = [(-1, -1), (140, 200), (40, 100), (180, 240), (280, 360)]
    densities = [2.3, 2.6, 2.4, 2.6, 2.8]
    ellipse_density = 2
    ellipse_axes_bound = [2, 10]
    dip_range = (-30, 30)
    throw_range = (1, 5)

    grid_depth = grid_size[0]

    gravity_matrix = generate_gravity_matrix()


    for _ in tqdm(range(num_models)):
        model = np.zeros(grid_size)

        # Generate layers
        current_layer = grid_depth
        for layer_number in [5, 4, 3, 2]:
            layer_index = layer_number - 1

            thickness = np.random.uniform(
                table_bounds[layer_index][0], table_bounds[layer_index][1]
            )
            cell_thickness = round(thickness / grid_depth)

            # Fill from current layer to curent_layer - cell_thickness
            model[current_layer - cell_thickness : current_layer, :] = densities[
                layer_index
            ]

            current_layer -= cell_thickness

        if current_layer > 0:
            model[:current_layer, :] = densities[0]  # Layer 1 fills remaining space



        if np.random.rand() < 0.7:
            ellipse_center = (
                np.random.randint(grid_size[0]),
                np.random.randint(grid_size[1]),
            )
            ellipse_axes = (
                np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]),
                np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]),
            )  # Example axis lengths

            model = add_ellipse(model, ellipse_center, ellipse_axes, ellipse_density)

        if np.random.rand() < 0.8:
            height, width = model.shape
            dip_degrees = np.random.uniform(*dip_range)
            throw = np.random.randint(*throw_range)
            fault_x = np.random.randint(width)
            model = add_fault(model, dip_degrees, throw, fault_x)

            fault_sum_up = [True, dip_degrees, throw, fault_x]

        else:
            fault_sum_up = [False, 0, 0, 0]

    
        gravity_measure = compute_gravity_measure(model, gravity_matrix)



        faults.append(fault_sum_up)
        
        gravity_measures.append(gravity_measure)
        

        models.append(model)

    if save_model:
        with h5py.File(path_save, "w") as f:
            # just concatenate all models and gravity measures and faults so that you can just save with the keys x for the models and y for the gravity measures and faults for the faults
            f.create_dataset("x", data=np.array(models))
            f.create_dataset("y", data=np.array(gravity_measures))
            f.create_dataset("faults", data=np.array(faults))

    return models, gravity_measures


def compute_gravity_anomaly(gamma, Y1, Y2, Z1, Z2):
    factor = (
        Y2 * np.log2((Y2**2 + Z2**2) / (Y2**2 + Z1**2))
        + Y1 * np.log2((Y1**2 + Z1**2) / (Y1**2 + Z2**2))
        + 2 * Z2 * (np.arctan(Y2 / Z2) - np.arctan(Y1 / Z2))
        + 2 * Z1 * (np.arctan(Y1 / Z1) - np.arctan(Y2 / Z1))
    )
    Gz = gamma * factor

    return Gz


def compute_gravity_measure(model, gravity_matrix):
    dense_model = np.repeat(model, 2, axis=1)
    # multiply elementwise gravity_matrix with dense_model and sum on dimensions 1 and 2
    gravity_data = []
    for i in range(32):
        gravity_data.append(np.sum(np.multiply(gravity_matrix[i], dense_model)))

    gravity_data -= gravity_data[0]

    return gravity_data



def generate_gravity_matrix( observer_height=50, cell_width=12.5, cell_height=25, grid_height=32, grid_width=64, n_measurement_points=32):

    measurement_points = np.linspace(212.5, 600, n_measurement_points)
    gravity_matrix = np.zeros((n_measurement_points, grid_height, grid_width))

    gamma = 0.0000000000667
    for i, point_y in enumerate(measurement_points):
        for col in range(grid_width):
            for row in range(grid_height):
                # Coordinates of the prism in meters
                Z1 = row * cell_height + observer_height
                Z2 = (row + 1) * cell_height + observer_height

                ya = abs((col + 1) * cell_width - point_y)
                yb = abs(col * cell_width - point_y)
                Y1 = min(ya, yb)
                Y2 = max(ya, yb)

                if Y1 <= 200 and Y2 <= 200:
                    gravity_matrix[i, row, col] = compute_gravity_anomaly(
                        gamma, Y1, Y2, Z1, Z2
                    )

    return  gravity_matrix




"""DRAFT - Visualization functions:"""
def plot_models(models):
    num_models = len(models)
    cols = 4
    rows = num_models // cols + (num_models % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for idx, model in enumerate(models):
        ax = axs.flatten()[idx]
        cax = ax.imshow(model, cmap="viridis")
        ax.axis("off")  # Hide grid lines and ticks
    fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.95)
    plt.show()


def plot_models_and_gravity_data(models, gravity_data_list):
    num_models = len(models)
    cols = 4  # Define number of columns for subplot grid
    rows = (
        num_models // cols * 2 + (num_models % cols > 0) * 2
    )  # Double the rows for models and gravity data
    fig, axs = plt.subplots(
        rows, cols, figsize=(15, 2 * rows)
    )  # Adjust figure size dynamically based on content

    for idx, (model, gravity_data) in enumerate(zip(models, gravity_data_list)):
        # Plot the model
        ax_model = axs[2 * (idx // cols), idx % cols]
        cax = ax_model.imshow(model, cmap="viridis")
        ax_model.axis("off")  # Hide grid lines and ticks

        # Plot the gravity data
        ax_gravity = axs[2 * (idx // cols) + 1, idx % cols]
        ax_gravity.plot(np.linspace(200, 600, 32), gravity_data, label="Gravity Data")
        ax_gravity.set_title("Model {}".format(idx + 1))
        ax_gravity.label_outer()  # Simplify axis labels for clarity

    fig.colorbar(
        cax, ax=axs[::2, :].ravel().tolist(), shrink=0.95
    )  # Add a single color bar for all model plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    
    # if the path exists:
    path_saving_antoine = "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/data/gravity_data/models_and_gravity_data.png"
    if os.path.exists(path_saving_antoine):
        plt.savefig(path_saving_antoine)
    
    plt.show()


if __name__ == "__main__":
    # num_models = 10000
    # models, all_gravity_data = generate_models(
    #     num_models,
    #     path_save="/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/data/gravity_data/train_data.hdf5",
    #     save_model=True,
    # )

    # print("Train set generated successfully!")

    num_models = 8
    models, all_gravity_data = generate_models(
        num_models,
        # path_save="/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/data/gravity_data/val_data.hdf5",
        # save_model=True,
    )

    # print("Val set generated successfully!")

    # open the saved models, gravity measures and faults and plot it
    # with h5py.File(
    #     "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/data/gravity_data/train_data.hdf5",
    #     "r",
    # ) as f:
    #     models = list(f["x"])
    #     all_gravity_data = list(f["y"])
    #     faults = list(f["faults"])
    # all_gravity_data = [
    #     compute_gravity_data(model) for model in models
    # ]  # Compute gravity data for each model

    # Plot the models and their corresponding gravity data
    plot_models_and_gravity_data(models, all_gravity_data)
