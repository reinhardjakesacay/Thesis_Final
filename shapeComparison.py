import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import scipy.stats as stats
import pandas as pd

def extract_gray_shape(image_path):
    """
    Extract gray shape region from an image, ignoring blue background.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        tuple: Mask of gray shape and original image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None, None
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    return mask, image

def calculate_iou_and_dice(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) and Dice Coefficient.
    
    Args:
        mask1 (numpy.ndarray): First binary mask
        mask2 (numpy.ndarray): Second binary mask
    
    Returns:
        tuple: IoU and Dice Coefficient
    """
    # Ensure masks are binary
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    # Intersection and union for IoU
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    
    # Dice Similarity Coefficient
    dice = (2 * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2)) if np.sum(mask1) + np.sum(mask2) != 0 else 0
    
    return iou, dice

def compute_model_differences(processed_folder, ca_folder, ca_rfa_folder, output_folder='batch_difference_results'):
    """
    Compute and analyze differences between model comparisons.
    
    Args:
        processed_folder (str): Path to processed model images
        ca_folder (str): Path to CA model images
        ca_rfa_folder (str): Path to CA-RFA model images
        output_folder (str, optional): Folder to save results
    
    Returns:
        dict: Comprehensive difference analysis results
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    processed_images = sorted([f for f in os.listdir(processed_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_images = sorted([f for f in os.listdir(ca_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_rfa_images = sorted([f for f in os.listdir(ca_rfa_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Ensure we have matching number of images
    min_images = min(len(processed_images), len(ca_images), len(ca_rfa_images))
    
    # Initialize storage for metrics
    ca_iou_values = []
    ca_dice_values = []
    ca_rfa_iou_values = []
    ca_rfa_dice_values = []
    
    # Compute metrics for each image
    for i in range(min_images):
        # Construct full image paths
        processed_path = os.path.join(processed_folder, processed_images[i])
        ca_path = os.path.join(ca_folder, ca_images[i])
        ca_rfa_path = os.path.join(ca_rfa_folder, ca_rfa_images[i])
        
        # Extract masks
        processed_mask, _ = extract_gray_shape(processed_path)
        ca_mask, _ = extract_gray_shape(ca_path)
        ca_rfa_mask, _ = extract_gray_shape(ca_rfa_path)
        
        # Skip if any mask is None
        if processed_mask is None or ca_mask is None or ca_rfa_mask is None:
            continue
        
        # Resize masks to a common shape
        base_shape = (512, 512)
        processed_mask = cv2.resize(processed_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        ca_mask = cv2.resize(ca_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        ca_rfa_mask = cv2.resize(ca_rfa_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate similarities
        ca_iou, ca_dice = calculate_iou_and_dice(processed_mask, ca_mask)
        ca_rfa_iou, ca_rfa_dice = calculate_iou_and_dice(processed_mask, ca_rfa_mask)
        
        # Store values
        ca_iou_values.append(ca_iou)
        ca_dice_values.append(ca_dice)
        ca_rfa_iou_values.append(ca_rfa_iou)
        ca_rfa_dice_values.append(ca_rfa_dice)
    
    # Compute comprehensive difference analysis
    difference_analysis = {
        # Mean (Average) Metrics
        'mean_ca_iou': np.mean(ca_iou_values),
        'mean_ca_dice': np.mean(ca_dice_values),
        'mean_ca_rfa_iou': np.mean(ca_rfa_iou_values),
        'mean_ca_rfa_dice': np.mean(ca_rfa_dice_values),
        
        # Absolute Difference Between Models
        'abs_mean_iou_difference': np.abs(np.mean(ca_iou_values) - np.mean(ca_rfa_iou_values)),
        'abs_mean_dice_difference': np.abs(np.mean(ca_dice_values) - np.mean(ca_rfa_dice_values)),
        
        # Variance of Metrics
        'variance_ca_iou': np.var(ca_iou_values),
        'variance_ca_rfa_iou': np.var(ca_rfa_iou_values),
        
        # Statistical Tests
        'iou_ttest': stats.ttest_ind(ca_iou_values, ca_rfa_iou_values),
        'dice_ttest': stats.ttest_ind(ca_dice_values, ca_rfa_dice_values),
        
        # Full Metrics for Detailed Analysis
        'ca_iou_values': ca_iou_values,
        'ca_dice_values': ca_dice_values,
        'ca_rfa_iou_values': ca_rfa_iou_values,
        'ca_rfa_dice_values': ca_rfa_dice_values
    }
    
    # Visualization of Differences
    plt.figure(figsize=(12, 5))
    
    # IoU Comparison
    plt.subplot(1, 2, 1)
    plt.boxplot([ca_iou_values, ca_rfa_iou_values], labels=['CA Model', 'CA-RFA Model'])
    plt.title('IoU Distribution Comparison')
    plt.ylabel('Intersection over Union (IoU)')
    
    # Dice Coefficient Comparison
    plt.subplot(1, 2, 2)
    plt.boxplot([ca_dice_values, ca_rfa_dice_values], labels=['CA Model', 'CA-RFA Model'])
    plt.title('Dice Coefficient Distribution Comparison')
    plt.ylabel('Dice Coefficient')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_comparison_boxplot.png'))
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'CA Model IoU': ca_iou_values,
        'CA Model Dice': ca_dice_values,
        'CA-RFA Model IoU': ca_rfa_iou_values,
        'CA-RFA Model Dice': ca_rfa_dice_values
    })
    results_df.to_csv(os.path.join(output_folder, 'model_difference_analysis.csv'), index=False)
    
    # Print summary to console
    print("\nModel Comparison Summary:")
    print(f"Average CA Model IoU: {difference_analysis['mean_ca_iou']:.4f}")
    print(f"Average CA-RFA Model IoU: {difference_analysis['mean_ca_rfa_iou']:.4f}")
    print(f"Absolute IoU Difference: {difference_analysis['abs_mean_iou_difference']:.4f}")
    print(f"T-Test for IoU: t-statistic = {difference_analysis['iou_ttest'].statistic:.4f}, p-value = {difference_analysis['iou_ttest'].pvalue:.4f}")
    
    return difference_analysis

def batch_compare_models(processed_folder, ca_folder, ca_rfa_folder, output_folder='batch_similarity_results'):
    """
    Compare batch of images from different model folders.
    
    Args:
        processed_folder (str): Path to processed model images
        ca_folder (str): Path to CA model images
        ca_rfa_folder (str): Path to CA-RFA model images
        output_folder (str, optional): Folder to save results
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'plots'), exist_ok=True)
    
    # Get list of image files
    processed_images = sorted([f for f in os.listdir(processed_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_images = sorted([f for f in os.listdir(ca_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_rfa_images = sorted([f for f in os.listdir(ca_rfa_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Ensure we have matching number of images
    min_images = min(len(processed_images), len(ca_images), len(ca_rfa_images))
    
    # Prepare CSV for results
    csv_filename = os.path.join(output_folder, 'batch_similarity_results.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'CA Model IoU', 'CA Model Dice', 'CA-RFA Model IoU', 'CA-RFA Model Dice'])
    
    # Batch comparison
    for i in range(min_images):
        # Construct full image paths
        processed_path = os.path.join(processed_folder, processed_images[i])
        ca_path = os.path.join(ca_folder, ca_images[i])
        ca_rfa_path = os.path.join(ca_rfa_folder, ca_rfa_images[i])
        
        # Extract masks
        processed_mask, processed_image = extract_gray_shape(processed_path)
        ca_mask, ca_image = extract_gray_shape(ca_path)
        ca_rfa_mask, ca_rfa_image = extract_gray_shape(ca_rfa_path)
        
        # Skip if any mask is None
        if processed_mask is None or ca_mask is None or ca_rfa_mask is None:
            print(f"Skipping image set {i+1} due to mask extraction failure")
            continue
        
        # Resize masks to a common shape
        base_shape = (512, 512)
        processed_mask = cv2.resize(processed_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        ca_mask = cv2.resize(ca_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        ca_rfa_mask = cv2.resize(ca_rfa_mask, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate similarities
        ca_iou, ca_dice = calculate_iou_and_dice(processed_mask, ca_mask)
        ca_rfa_iou, ca_rfa_dice = calculate_iou_and_dice(processed_mask, ca_rfa_mask)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        overlay_color = (0, 255, 0)  # Green overlay
        alpha_value = 0.5
        
        # Plot processed model
        axes[0].imshow(processed_image)
        processed_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_filled = np.zeros_like(processed_image)
        cv2.drawContours(processed_filled, processed_contours, -1, overlay_color, thickness=cv2.FILLED)
        axes[0].imshow(processed_filled, alpha=alpha_value)
        axes[0].set_title(f"Processed Model\n{processed_images[i]}")
        axes[0].axis('off')
        
        # Plot CA model
        axes[1].imshow(ca_image)
        ca_contours, _ = cv2.findContours(ca_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ca_filled = np.zeros_like(ca_image)
        cv2.drawContours(ca_filled, ca_contours, -1, overlay_color, thickness=cv2.FILLED)
        axes[1].imshow(ca_filled, alpha=alpha_value)
        axes[1].set_title(f"CA Model\nIoU: {ca_iou:.4f}\nDice: {ca_dice:.4f}")
        axes[1].axis('off')
        
        # Plot CA-RFA model
        axes[2].imshow(ca_rfa_image)
        ca_rfa_contours, _ = cv2.findContours(ca_rfa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ca_rfa_filled = np.zeros_like(ca_rfa_image)
        cv2.drawContours(ca_rfa_filled, ca_rfa_contours, -1, overlay_color, thickness=cv2.FILLED)
        axes[2].imshow(ca_rfa_filled, alpha=alpha_value)
        axes[2].set_title(f"CA-RFA Model\nIoU: {ca_rfa_iou:.4f}\nDice: {ca_rfa_dice:.4f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(output_folder, 'plots', f'batch_comparison_{i+1}.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Append results to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                processed_images[i],
                ca_iou, ca_dice,
                ca_rfa_iou, ca_rfa_dice
            ])
        
        print(f"Processed image set {i+1}")
    
    # After batch processing, call the difference analysis
    difference_results = compute_model_differences(processed_folder, ca_folder, ca_rfa_folder, output_folder)
    
    print(f"Batch comparison complete. Results saved in {output_folder}")
    return difference_results

# Example usage
if __name__ == "__main__":
    # Paths to your specific folders
    processed_folder = "./images_processed_typhoon"
    ca_folder = "./images_Reg_CA_model"
    ca_rfa_folder = "./images_Hybrid_Model"
    
    # Run batch comparison and get difference analysis
    difference_analysis = batch_compare_models(processed_folder, ca_folder, ca_rfa_folder)