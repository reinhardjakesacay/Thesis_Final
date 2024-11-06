import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    "CA_RFA_model": r"images_Hybrid_Model/Hybrid_Model_RFA_23.png",  # Model 1
    "processed_model": r"images_processed_typhoon/processed_storm_track_1.png",  # Model 2
    "CA_model": r"images_Reg_CA_model/Reg_CA_Model_23.png"  # Model 3
}

# Extract gray shape regions from each model image and the original images
gray_masks = {}
original_images = {}
for name, path in image_paths.items():
    mask, image = extract_gray_shape(path)
    gray_masks[name] = mask
    original_images[name] = image

# Resize to base shape (common shape for comparison)
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
# We choose the model with the highest IoU or Dice coefficient
best_model = max(similarity_results, key=lambda x: similarity_results[x]['IoU'])  # Or 'Dice Coefficient'
print(f"The model closest to the processed typhoon path is: {best_model}")

# Plotting the images and masks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, model_name in enumerate(image_paths.keys()):
    ax = axes[i]
    ax.imshow(original_images[model_name])
    
    # Draw the contours or the mask on the image
    contours, _ = cv2.findContours(gray_masks[model_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ax.imshow(cv2.drawContours(np.copy(original_images[model_name]), contours, -1, (0, 255, 0), 2), alpha=0.6)
    
    # Title with similarity score
    similarity_score = similarity_results.get(model_name, {})
    ax.set_title(f"{model_name}\nIoU: {similarity_score.get('IoU', 'N/A')}\nDice: {similarity_score.get('Dice Coefficient', 'N/A')}", fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()
