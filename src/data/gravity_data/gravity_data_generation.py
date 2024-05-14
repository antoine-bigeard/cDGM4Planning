import numpy as np
import matplotlib.pyplot as plt

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


def add_fault(model, throw_bound):
    """
    Adds a fault to the model.

    Args:
        model (numpy.ndarray): The model to add the fault to.
        throw_bound (tuple): The range of the throw.

    Returns:
        numpy.ndarray: The updated model with the fault added.
    """
    fault_line = np.random.randint(model.shape[1])  # Avoid the very first and last column
    throw = np.random.randint(-5, 5)  # Smaller throw for better control and visual effect
    dip = np.random.randint(-45, 45)  # Lower dip for better control and visual effect

    return model

def generate_models(num_models=10000, grid_size=(32, 32)):
    """
    Generates a specified number of gravity models with random layer thicknesses and densities.
    
    Args:
        num_models (int, optional): The number of models to generate. Defaults to 10000.
        grid_size (tuple, optional): The size of the grid for each model. Defaults to (32, 32).
    
    Returns:
        list: A list of generated models, where each model is a 2D numpy array representing the density of each grid cell.
    """

    models = []
    table_bounds = [(-1, -1), (140, 200), (40, 100), (180, 240), (280, 360)]
    densities = [2.3, 2.6, 2.4, 2.6, 2.8]
    ellipse_density = 2
    ellipse_axes_bound = [2, 10]
    throw_bound = [-5, 5]

    grid_depth = grid_size[0]
    
    for _ in range(num_models):
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

        # Add anomalies and faults if applicable
        if np.random.rand() < 0.7:
            ellipse_center = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
            ellipse_axes = (np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]), np.random.randint(ellipse_axes_bound[0], ellipse_axes_bound[1]))  # Example axis lengths
            
            model = add_ellipse(model, ellipse_center, ellipse_axes, ellipse_density)
        
        # if np.random.rand() < 0.8:
        #     model = add_fault(model, throw_bound)
        
        models.append(model)
    
    return models



def gravity_anomaly(prism_density, Y1, Y2, Z1, Z2):
    gamma = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    
    factor = (
        (Y2 * np.log2((Y2**2 + Z2**2) / (Y2**2 + Z1**2)) + 
         Y1 * np.log2((Y1**2 + Z1**2) / (Y1**2 + Z2**2)) +
         2 * Z2 * (np.arctan(Y2 / Z2) - np.arctan(Y1 / Z2)) +
         2 * Z1 * (np.arctan(Y1 / Z1) - np.arctan(Y2 / Z1)))
    )
    Gz = gamma * prism_density * factor

    # print(factor, Y1, Y2, Z1, Z2)

    return Gz


def compute_gravity_data(model):
    cell_width = 25  # each cell is 25m
    grid_size = model.shape[0]
    measurement_points = np.linspace(200, 600, 33)
    observer_height = 50
    gravity_data = np.zeros(len(measurement_points))

    # Compute gravity at each measurement point
    for i, point_y in enumerate(measurement_points):
        for col in range(grid_size):
            for row in range(grid_size):
                # Coordinates of the prism in meters
                Z1 = row * cell_width  + observer_height
                Z2 = (row + 1) * cell_width + observer_height

                ya = abs((col + 1) * cell_width - point_y)
                yb = abs(col * cell_width - point_y)
                Y1 = min(ya, yb)
                Y2 = max(ya, yb)

                # Add the gravity effect of the current cell
                # gravity_data[i] += gravity_anomaly(model[row][col], Y1, Y2, Z1, Z2)

                if Y1 <= 200 and Y2 <= 200:
                    gravity_data[i] += gravity_anomaly(model[row][col], Y1, Y2, Z1, Z2)                

                
    # Normalize gravity data
    gravity_data -= gravity_data[0]
    
    return gravity_data


# Visualization function
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
        ax_gravity.plot(np.linspace(200, 600, 17), gravity_data, label='Gravity Data')
        ax_gravity.set_title('Model {}'.format(idx + 1))
        ax_gravity.label_outer()  # Simplify axis labels for clarity

    fig.colorbar(cax, ax=axs[::2, :].ravel().tolist(), shrink=0.95)  # Add a single color bar for all model plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

if __name__ == "__main__":
    num_models = 8
    models = generate_models(num_models)  # Generate 32 models
    all_gravity_data = [compute_gravity_data(model) for model in models]  # Compute gravity data for each model

    # Plot the models and their corresponding gravity data
    plot_models_and_gravity_data(models, all_gravity_data)