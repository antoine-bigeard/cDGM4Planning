import numpy as np
import matplotlib.pyplot as plt
import os
import json
import yaml

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
            if (x - center[0])**2 / axes[0]**2 + (y - center[1])**2 / axes[1]**2 <= 1:
                model[x, y] = density

    return model


def add_fault(model, throw, dip, horizontal_pos):
    """
    Adds a fault to the model.

    Args:
        model (numpy.ndarray): The model to add the fault to.
        throw (int): The vertical displacement of the fault.
        dip (float, optional): The angle of the fault in degrees. Defaults to a random angle between 60 and 120 degrees.
        horizontal_pos (int, optional): The horizontal position of the fault. If None, it will be randomly chosen.

    Returns:
        numpy.ndarray: The updated model with the fault added.
    """
    return model

def gravity_anomaly(prism_density: float, Y1: float, Y2: float, Z1: float, Z2: float) -> float:
    """
    Computes the gravity anomaly from a prism with the given density, and vertices (Y1, Y2, Z1, Z2)

    Args:
        prism_density (float): The density of the prism.
        Y1 (float): The y-coordinate of the first vertex of the prism.
        Y2 (float): The y-coordinate of the second vertex of the prism.
        Z1 (float): The z-coordinate of the first vertex of the prism.
        Z2 (float): The z-coordinate of the second vertex of the prism.

    Returns:
        float: The computed gravity anomaly.
    """
    # Constants
    gamma = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

    # Compute the factor of the gravity anomaly
    factor = (
        (Y2 * np.log2((Y2**2 + Z2**2) / (Y2**2 + Z1**2)) + 
         Y1 * np.log2((Y1**2 + Z1**2) / (Y1**2 + Z2**2)) +
         2 * Z2 * (np.arctan(Y2 / Z2) - np.arctan(Y1 / Z2)) +
         2 * Z1 * (np.arctan(Y1 / Z1) - np.arctan(Y2 / Z1)))
    )

    # Compute the gravity anomaly
    Gz = gamma * prism_density * factor

    return Gz



def compute_gravity_data(model):
    cell_width = 25  # each cell is 25m
    grid_size = model.shape[0]
    measurement_points = np.linspace(200, 600, 33)
    observer_height = 50
    gravity_data = np.zeros(len(measurement_points))

    # Compute gravity at each measurement point
    for i, point_y in enumerate(measurement_points):
        if point_y % 25 == 0:
            idx = int(point_y//25) # We are in between the cell number idx and idx+1 (accessed ny idx-1 and idx in the 0-index matrix)
            col_range = list(range(idx-8, idx+8))

        elif (point_y *2) % 25 == 0:
            idx_minus = int((point_y-12.5)//25)
            col_range = list(range(idx_minus - 7, idx_minus+1)) + list(range(idx_minus, idx_minus + 8)) # We count twice the cell which is just below the point, corresponding to idx_minus
        

        else:
            raise ValueError("Number of measurement points should be either 16+1 or 32+1")
        
        dist = [-175, -150, -125, -100, -75, -50, -25, 0, 0, 25, 50, 75, 100, 125, 150, 175]
        for j, col_idx in enumerate(col_range):
            for row_idx in range(grid_size):
                # Coordinates of the prism in meters
                Z1 = row_idx * cell_width  + observer_height
                Z2 = (row_idx + 1) * cell_width + observer_height

                Y1 = dist[j]
                Y2 = dist[j] + 25
                
                gravity_data[i] += gravity_anomaly(model[row_idx][col_idx], Y1, Y2, Z1, Z2)                
 
    # Normalize gravity data
    gravity_data -= gravity_data[0]
    
    return gravity_data



def generate_models(num_models=10000, grid_size=(32, 32)):
    """
    Generates a specified number of gravity models with random layer thicknesses and densities.
    
    Args:
        num_models (int, optional): The number of models to generate. Defaults to 10000.
        grid_size (tuple, optional): The size of the grid for each model. Defaults to (32, 32).
    
    Returns:
        list: A list of generated models, where each model is a 2D numpy array representing the density of each grid cell.
    """

    models, gravity_conditions, params = [], [], []
    table_bounds = [(-1, -1), (140, 200), (40, 100), (180, 240), (280, 360)]
    densities = [2.3, 2.6, 2.4, 2.6, 2.8]
    ellipse_density = 2
    ellipse_axes_bound = [2, 10]

    grid_depth = grid_size[0]
    
    for i in range(num_models):
        model = np.zeros(grid_size)
        
        # Generate layers
        current_layer = grid_depth
        for layer_number in [5, 4, 3, 2]:
            layer_index = layer_number - 1

            thickness = np.random.uniform(table_bounds[layer_index][0], table_bounds[layer_index][1])
            cell_thickness = round(thickness / grid_depth)

            # Fill from current layer to curent_layer - cell_thickness
            model[current_layer - cell_thickness:current_layer, :] = densities[layer_index] 

            current_layer -= cell_thickness
        
        if current_layer > 0:
            model[:current_layer, :] = densities[0] # Layer 1 fills remaining space


        # TODO: Add faults
        if np.random.rand() < 0.8:
            throw = np.random.randint(-5, 5)
            dip = np.random.uniform(60, 120)
            horizontal_pos = np.random.randint(0, model.shape[1])
    
            model = add_fault(model, throw, dip, horizontal_pos)
        else:
            throw, dip, horizontal_pos = None, None, None

        if np.random.rand() < 0.7:
            ellipse_center = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
            ellipse_axes = (np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]), np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]))  # Example axis lengths
            
            model = add_ellipse(model, ellipse_center, ellipse_axes, ellipse_density)
        else:
            ellipse_center, ellipse_axes = None, None
        


        # Generate the associated gravity condition 
        associated_gravity_condition = compute_gravity_data(model)

        # Normalize our model:
        # rho_min, rho_max = 2, 2.8
        # normalized_model = (model - rho_min) / (rho_max - rho_min) * 1.8 - 0.9
        normalized_model = model       

        param = {
            "idx": i,
            "elipse_center": ellipse_center,
            "elipse_axes": ellipse_axes,
            "throw": throw,
            "dip": dip,
            "horizontal_pos": horizontal_pos,
        }

        models.append(normalized_model)
        gravity_conditions.append(associated_gravity_condition)
        params.append(param)

    
    return models, gravity_conditions, params



def create_dataset(params):

    print("Starting dataset creation with params: ", params)

    try:
        train_length, val_length, test_length = params["train_length"], params["val_length"], params["test_length"]
        data_base_dir = params["data_base_dir"]
    except:
        raise ValueError("Params file incomplete.")
    


    # Create the train, val and test dir if they don't exist :
    train_folder, val_folder, test_folder = data_base_dir + "train/", data_base_dir + "val/", data_base_dir + "test/"
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


    nbr_samples_to_generate = train_length + val_length + test_length
    models, gravity_conditions, params= generate_models(nbr_samples_to_generate)
    
    # Write the params to a json file in the main folder:
    with open(data_base_dir + "params.json", "w") as f:
        json.dump(params, f)
    
    # Wrte the models to train/matrices, val/matrices and test/matrices, and the associated gravity conditions to train/gravity_conditions, val/gravity_conditions and test/gravity_conditions
    for i in range(nbr_samples_to_generate):
        if i < train_length:
            np.save(train_folder + str(i) + "_matrix" + ".npy", models[i])
            np.save(train_folder + + str(i) + "_gravity_condition" + ".npy", gravity_conditions[i])
        elif i < train_length + val_length:
            np.save(val_folder + str(i) + "_matrix" + ".npy", models[i])
            np.save(val_folder + str(i) + "_gravity_condition" + ".npy", gravity_conditions[i])
        else:
            np.save(test_folder + str(i) + "_matrix" + ".npy", models[i])
            np.save(test_folder + str(i) + "_gravity_condition" + ".npy", gravity_conditions[i])

    print("Dataset creation successfully completed! France. Origine France.")




"""DRAFT - Visualization functions:"""
def plot_models(models):
    num_models = len(models)
    cols = 4
    rows = num_models // cols + (num_models % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    
    for idx, model in enumerate(models):
        ax = axs.flatten()[idx]
        cax = ax.imshow(model, cmap='viridis')
        ax.axis('off')  # Hide grid lines and ticks
    fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.95)
    plt.show()


def plot_models_and_gravity_data(models, gravity_data_list):
    num_models = len(models)
    cols = 4  # Define number of columns for subplot grid
    rows = num_models // cols * 2 + (num_models % cols > 0) * 2  # Double the rows for models and gravity data
    fig, axs = plt.subplots(rows, cols, figsize=(15, 2 * rows))  # Adjust figure size dynamically based on content

    for idx, (model, gravity_data) in enumerate(zip(models, gravity_data_list)):
        # Plot the model
        ax_model = axs[2 * (idx // cols), idx % cols]
        cax = ax_model.imshow(model, cmap='viridis')
        ax_model.axis('off')  # Hide grid lines and ticks

        # Plot the gravity data
        ax_gravity = axs[2 * (idx // cols) + 1, idx % cols]
        ax_gravity.plot(np.linspace(200, 600, 33), gravity_data, label='Gravity Data')
        ax_gravity.set_title('Model {}'.format(idx + 1))
        ax_gravity.label_outer()  # Simplify axis labels for clarity

    fig.colorbar(cax, ax=axs[::2, :].ravel().tolist(), shrink=0.95)  # Add a single color bar for all model plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

if __name__ == "__main__":
    # params_json_path = "src\data\gravity_data\gravity_data_generation_params.yaml"
    # with open(params_json_path, "r") as f:
    #     params = yaml.safe_load(f)

    # create_dataset(params)



    num_models = 8
    models, _, _ = generate_models(num_models)  # Generate 32 models
    all_gravity_data = [compute_gravity_data(model) for model in models]  # Compute gravity data for each model

    # Plot the models and their corresponding gravity data
    plot_models_and_gravity_data(models, all_gravity_data)