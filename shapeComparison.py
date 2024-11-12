import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Helper function to extract gray shape region from image (ignores blue background)
def extract_gray_shape(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    return mask, image

# Example dictionary of image paths for the models
image_paths = {
    "CA_RFA_model": r"images_Hybrid_Model/Hybrid_Model_RFA_100.png",
    "processed_model": r"images_processed_typhoon/processed_storm_track_1.png",
    "CA_model": r"images_Reg_CA_model/Reg_CA_Model_100.png"
}

# Extract gray shape regions and original images
gray_masks = {}
original_images = {}
for name, path in image_paths.items():
    mask, image = extract_gray_shape(path)
    gray_masks[name] = mask
    original_images[name] = image

# Resize masks to a common shape (base_shape)
base_shape = (512, 512)
for name in gray_masks:
    if gray_masks[name].shape != base_shape:
        gray_masks[name] = cv2.resize(gray_masks[name], (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)

# Function to calculate IoU and Dice similarity
def calculate_iou_and_dice(mask1, mask2):
    # Intersection and union for IoU
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    
    # Dice Similarity Coefficient
    dice = (2 * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2)) if np.sum(mask1) + np.sum(mask2) != 0 else 0
    
    return iou, dice

# Compare each model (CA_RFA_model, CA_model) to the processed model
similarity_results = {}
for model_name, mask in gray_masks.items():
    if model_name != "processed_model":  # Compare only the first two models to the processed model
        iou, dice = calculate_iou_and_dice(gray_masks["processed_model"], mask)
        similarity_results[model_name] = {
            "IoU": round(iou, 4),
            "Dice Coefficient": round(dice, 4)
        }

# Print the similarity results
print("Similarity Results:")
for model_name, metrics in similarity_results.items():
    print(f"{model_name}: IoU = {metrics['IoU']}, Dice Coefficient = {metrics['Dice Coefficient']}")

# Determine which model is closer to the processed typhoon path
best_model = max(similarity_results, key=lambda x: similarity_results[x]['IoU'])
print(f"The model closest to the processed typhoon path is: {best_model}")

# Plotting the images and masks with consistent color overlay
fig, axes = plt.subplots(1, 3, figsize=(10, 4))  # Smaller figure size for compact display

overlay_color = (0, 255, 0)  # Green overlay color for filled shapes
alpha_value = 0.5  # Consistent alpha for both CA and CA-RFA models

for i, model_name in enumerate(image_paths.keys()):
    ax = axes[i]
    ax.imshow(original_images[model_name])
    
    # Draw filled contours with thicker lines for more visibility
    contours, _ = cv2.findContours(gray_masks[model_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(original_images[model_name])
    
    # Increase thickness for better visibility
    cv2.drawContours(filled_image, contours, -1, overlay_color, thickness=cv2.FILLED)  # Filled contour
    
    # Optionally apply a dilation to the mask to increase coverage
    dilated_mask = cv2.dilate(gray_masks[model_name], np.ones((10, 10), np.uint8), iterations=1)
    cv2.drawContours(filled_image, contours, -1, overlay_color, thickness=cv2.FILLED)
    ax.imshow(filled_image, alpha=alpha_value)  # Set consistent alpha for both models

    # Title with similarity score if available, skip for processed model
    if model_name != "processed_model":
        similarity_score = similarity_results.get(model_name, {})
        title = f"{model_name}\nIoU: {similarity_score.get('IoU', 'N/A')}\nDice: {similarity_score.get('Dice Coefficient', 'N/A')}"
    else:
        title = model_name  # Only show the model name for the processed model
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()

# Define the folder path where the plot will be saved
folder_path = 'images_similarity_result'  # Change this to your desired folder path

# Ensure the folder exists; if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate a unique filename for saving the plot
base_filename = 'similarity_result_1.png'
output_path = os.path.join(folder_path, base_filename)
counter = 1

# Check if the file already exists and generate a new filename if necessary
while os.path.exists(output_path):
    output_path = os.path.join(folder_path, f'similarity_result_{counter}.png')  # Include folder_path here
    counter += 1

# Save the plot to a file
plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save with tight bounding box and high resolution

    
# Open the existing similarity_results.csv file and append the results in the required format
csv_filename = "similarity_results.csv"

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Append data in the specified format
    writer.writerow([
        similarity_results["CA_model"]["IoU"],
        similarity_results["CA_model"]["Dice Coefficient"],
        similarity_results["CA_RFA_model"]["IoU"],
        similarity_results["CA_RFA_model"]["Dice Coefficient"]
    ])
print(f"Similarity results appended to {csv_filename}")

plt.show()
