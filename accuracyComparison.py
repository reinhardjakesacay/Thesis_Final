import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to calculate accuracy percentage
def calculate_accuracy(prediction_img, actual_img):
    # Ensure the images are the same size
    assert prediction_img.shape == actual_img.shape, "Images must be the same size for accuracy calculation."
    
    # Calculate the number of matching pixels
    correct_predictions = np.sum((prediction_img == actual_img).all(axis=2))
    total_pixels = prediction_img.shape[0] * prediction_img.shape[1]
    
    # Calculate accuracy percentage
    accuracy = (correct_predictions / total_pixels) * 100
    return accuracy

imageFileNum = input("Enter the number of model to compare: ")
processedTyphoon = input("Enter the number of the typhoon data: ")

# Read the images
ca_img = cv2.imread(f'images_Reg_CA_model\Reg_CA_Model_{imageFileNum}.png')
rfa_img = cv2.imread(f'images_Hybrid_Model\Hybrid_Model_RFA_{imageFileNum}.png')
actual_img = cv2.imread(f'images_processed_typhoon\processed_storm_track_{processedTyphoon}.png')

# Convert images to RGB for Matplotlib
ca_img = cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB)
rfa_img = cv2.cvtColor(rfa_img, cv2.COLOR_BGR2RGB)
actual_img = cv2.cvtColor(actual_img, cv2.COLOR_BGR2RGB)

# Ensure all images have the same size
height, width = actual_img.shape[:2]
ca_img = cv2.resize(ca_img, (width, height))
rfa_img = cv2.resize(rfa_img, (width, height))

# Create a blank image to overlay the predictions with the same size as the actual image
overlay_img = np.zeros_like(actual_img)

# Overlay the predictions with different colors
overlay_img[np.where((ca_img != 0).all(axis=2))] = [255, 50, 50]  # Bright red for CA
overlay_img[np.where((rfa_img != 0).all(axis=2))] = [255, 255, 50]  # Bright yellow for CA-RFA

# Combine the overlay with the actual storm track
combined_img = cv2.addWeighted(actual_img, 0.4, overlay_img, 0.6, 0)

# Calculate accuracy for CA and CA-RFA
ca_accuracy = calculate_accuracy(ca_img, actual_img)
rfa_accuracy = calculate_accuracy(rfa_img, actual_img)

# Determine which model is more accurate
more_accurate = "CA" if ca_accuracy > rfa_accuracy else "CA-RFA" if rfa_accuracy > ca_accuracy else "Both are equally accurate"

# Display the combined image
plt.figure(figsize=(10, 10))
plt.imshow(combined_img)
plt.axis('off')
plt.title('Enhanced Comparison of CA, CA-RFA, and Actual Storm Track')

# Create a custom legend
legend_labels = ['CA Prediction', 'CA-RFA Prediction', 'Actual Storm Track']
colors = [[255, 50, 50], [255, 255, 50], [50, 255, 50]]
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                        markerfacecolor=np.array(color)/255.0, markersize=10) 
           for label, color in zip(legend_labels, colors)]

plt.legend(handles=handles, loc='upper right', fontsize=12)

# Add accuracy results to the plot
plt.text(10, 20, f"CA Accuracy: {ca_accuracy:.2f}%", color='white', fontsize=12)
plt.text(10, 40, f"CA-RFA Accuracy: {rfa_accuracy:.2f}%", color='white', fontsize=12)
plt.text(10, 60, f"More Accurate Model: {more_accurate}", color='white', fontsize=12)

# Define the folder path where the plot will be saved
folder_path = 'images_comparison_result'  # Change this to your desired folder path

# Ensure the folder exists; if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate a unique filename for saving the plot
base_filename = 'comparison_result_1.png'
output_path = os.path.join(folder_path, base_filename)
counter = 1

# Check if the file already exists and generate a new filename if necessary
while os.path.exists(output_path):
    output_path = os.path.join(folder_path, f'comparison_result_{counter}.png')  # Include folder_path here
    counter += 1

# Save the plot to a file
plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save with tight bounding box and high resolution
plt.show()
