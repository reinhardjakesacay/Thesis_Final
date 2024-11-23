import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Load the dataset from a CSV file
csv_file = 'Plastic Waste Around the World.csv'  # Replace with your dataset file name
data = pd.read_csv(csv_file)

# Calculate the min and max values for each variable
total_plastic_min = data['Total_Plastic_Waste_MT'].min()
total_plastic_max = data['Total_Plastic_Waste_MT'].max()

recycling_rate_min = data['Recycling_Rate'].min()
recycling_rate_max = data['Recycling_Rate'].max()

per_capita_waste_min = data['Per_Capita_Waste_KG'].min()
per_capita_waste_max = data['Per_Capita_Waste_KG'].max()

# Define the folder path where the image will be saved
folder_path = 'images_Plastic_CA_model'

# Ensure the directory exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Function to run a single simulation
def run_single_simulation():
    # CA Simulation Parameters with randomized values based on dataset ranges
    grid_size = 200  # Randomized grid size
    steps = 100  # Randomized number of steps
    
    # Randomized parameters from dataset ranges
    total_plastic_waste_mt = np.random.uniform(total_plastic_min, total_plastic_max)
    recycling_rate = np.random.uniform(recycling_rate_min, recycling_rate_max)
    per_capita_waste_kg = np.random.uniform(per_capita_waste_min, per_capita_waste_max)

    # Categorize into high, medium, or low for waste spread probabilities
    if recycling_rate > 0.030 * recycling_rate_max:
        waste_prob = 0.10  # High recycling, lower spread probability
    elif recycling_rate > 0.164 * recycling_rate_max:
        waste_prob = 0.20  # Medium recycling, medium spread probability
    else:
        waste_prob = 0.30  # Low recycling, higher spread probability

    # Categorize max branches per cell
    if total_plastic_waste_mt > 0.012 * total_plastic_max:
        max_branches_per_cell = 5  # High waste, more branches
    elif total_plastic_waste_mt > 0.006 * total_plastic_max:
        max_branches_per_cell = 4  # Medium waste, medium branches
    else:
        max_branches_per_cell = 3  # Low waste, fewer branches

    # Categorize max distance
    if per_capita_waste_kg > 0.012 * per_capita_waste_max:
        max_distance = 5  # High per capita waste, larger spread
    elif per_capita_waste_kg > 0.036 * per_capita_waste_max:
        max_distance = 4  # Medium per capita waste, medium spread
    else:
        max_distance = 3  # Low per capita waste, smaller spread

    # Create initial grid and waste conditions
    plastic_grid = np.zeros((grid_size, grid_size))
    plastic_start_x = grid_size // 2
    plastic_start_y = grid_size // 2
    plastic_grid[plastic_start_x, plastic_start_y] = 1

    # Function to update the plastic blob
    def update_plastic(grid, blob_cells):
        new_grid = grid.copy()
        new_blob_cells = set(blob_cells)

        for (bx, by) in blob_cells:
            num_branches = np.random.randint(1, max_branches_per_cell + 1)
            for _ in range(num_branches):
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.randint(1, max_distance + 1)
                x_offset = int(distance * np.cos(angle))
                y_offset = int(distance * np.sin(angle))
                nx = bx + x_offset
                ny = by + y_offset

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if new_grid[nx, ny] == 0:
                        if np.random.rand() < waste_prob:
                            new_grid[nx, ny] = 1
                            new_blob_cells.add((nx, ny))

        return new_grid, new_blob_cells

    # Set up visualization
    cmap = colors.ListedColormap(['blue', 'gray'])
    norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    blob_cells = set([(plastic_start_x, plastic_start_y)])

    # Run the simulation for the determined number of steps
    for step in range(steps):
        plastic_grid, blob_cells = update_plastic(plastic_grid, blob_cells)
        ax.clear()
        ax.imshow(plastic_grid, cmap=cmap, norm=norm)
        ax.axis('off')
        plt.pause(0.05)  # Visualization pause for animation effect

    # Generate a unique filename for saving the plot
    base_filename = 'Plastic_CA_model_1.png'
    output_path = os.path.join(folder_path, base_filename)
    counter = 1

    # Check if the file already exists and generate a new filename if necessary
    while os.path.exists(output_path):
        output_path = os.path.join(folder_path, f'Plastic_CA_model_{counter}.png')
        counter += 1

    # Save the final picture
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Simulation saved at: {output_path}")

# Run a single simulation
run_single_simulation()
