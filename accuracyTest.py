import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def calculate_accuracy(prediction_img, actual_img):
    """
    Calculate pixel-wise accuracy between prediction and actual images.
    
    Args:
        prediction_img (numpy.ndarray): Predicted image
        actual_img (numpy.ndarray): Ground truth image
    
    Returns:
        float: Accuracy percentage
    """
    assert prediction_img.shape == actual_img.shape, "Images must be the same size for accuracy calculation."
    correct_predictions = np.sum((prediction_img == actual_img).all(axis=2))
    total_pixels = prediction_img.shape[0] * prediction_img.shape[1]
    accuracy = (correct_predictions / total_pixels) * 100
    return accuracy

def batch_compare_models(processed_folder, ca_folder, ca_rfa_folder, output_folder='batch_comparison_results'):
    """
    Batch compare multiple model predictions against actual images.
    
    Args:
        processed_folder (str): Path to processed/actual images
        ca_folder (str): Path to CA model predictions
        ca_rfa_folder (str): Path to CA-RFA model predictions
        output_folder (str, optional): Folder to save results
    
    Returns:
        dict: Summary of model performance
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get sorted list of images
    processed_images = sorted([f for f in os.listdir(processed_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_images = sorted([f for f in os.listdir(ca_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ca_rfa_images = sorted([f for f in os.listdir(ca_rfa_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Ensure we have matching number of images
    min_images = min(len(processed_images), len(ca_images), len(ca_rfa_images))
    
    # Initialize lists to store accuracies
    ca_accuracies = []
    ca_rfa_accuracies = []
    
    # Prepare empty list to store row dictionaries
    results_rows = []
    
    # Process each image set
    for i in range(min_images):
        # Read images
        actual_img = cv2.imread(os.path.join(processed_folder, processed_images[i]))
        ca_img = cv2.imread(os.path.join(ca_folder, ca_images[i]))
        ca_rfa_img = cv2.imread(os.path.join(ca_rfa_folder, ca_rfa_images[i]))
        
        # Convert to RGB
        actual_img = cv2.cvtColor(actual_img, cv2.COLOR_BGR2RGB)
        ca_img = cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB)
        ca_rfa_img = cv2.cvtColor(ca_rfa_img, cv2.COLOR_BGR2RGB)
        
        # Ensure all images have the same size
        height, width = actual_img.shape[:2]
        ca_img = cv2.resize(ca_img, (width, height))
        ca_rfa_img = cv2.resize(ca_rfa_img, (width, height))
        
        # Calculate accuracies
        ca_accuracy = calculate_accuracy(ca_img, actual_img)
        ca_rfa_accuracy = calculate_accuracy(ca_rfa_img, actual_img)
        
        # Store accuracies
        ca_accuracies.append(ca_accuracy)
        ca_rfa_accuracies.append(ca_rfa_accuracy)
        
        # Create overlay visualization
        overlay_img = np.zeros_like(actual_img)
        overlay_img[np.where((ca_img != 0).all(axis=2))] = [255, 50, 50]  # Bright red for CA
        overlay_img[np.where((ca_rfa_img != 0).all(axis=2))] = [255, 255, 50]  # Bright yellow for CA-RFA
        combined_img = cv2.addWeighted(actual_img, 0.4, overlay_img, 0.6, 0)
        
        # Save individual comparison plot
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f'Comparison {i+1}: CA vs CA-RFA')
        plt.text(10, 20, f"CA Accuracy: {ca_accuracy:.2f}%", color='white', fontsize=12)
        plt.text(10, 40, f"CA-RFA Accuracy: {ca_rfa_accuracy:.2f}%", color='white', fontsize=12)
        
        # Save individual plot
        individual_plot_path = os.path.join(output_folder, f'comparison_result_{i+1}.png')
        plt.savefig(individual_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Add to results list
        results_rows.append({
            'Image': processed_images[i],
            'CA Accuracy (%)': ca_accuracy,
            'CA-RFA Accuracy (%)': ca_rfa_accuracy
        })
    
    # Create DataFrame from results list
    results_csv_path = os.path.join(output_folder, 'batch_accuracy_results.csv')
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv_path, index=False)
    
    # Compute summary statistics
    summary = {
        'CA Model': {
            'Mean Accuracy (%)': np.mean(ca_accuracies),
            'Std Deviation (%)': np.std(ca_accuracies),
            'Min Accuracy (%)': np.min(ca_accuracies),
            'Max Accuracy (%)': np.max(ca_accuracies)
        },
        'CA-RFA Model': {
            'Mean Accuracy (%)': np.mean(ca_rfa_accuracies),
            'Std Deviation (%)': np.std(ca_rfa_accuracies),
            'Min Accuracy (%)': np.min(ca_rfa_accuracies),
            'Max Accuracy (%)': np.max(ca_rfa_accuracies)
        }
    }
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([ca_accuracies, ca_rfa_accuracies], labels=['CA Model', 'CA-RFA Model'])
    plt.title('Accuracy Distribution Comparison')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(output_folder, 'accuracy_distribution.png'))
    plt.close()
    
    # Print summary to console
    print("\nModel Comparison Summary:")
    print("CA Model:")
    print(f"  Mean Accuracy: {summary['CA Model']['Mean Accuracy (%)']:.2f}%")
    print(f"  Standard Deviation: {summary['CA Model']['Std Deviation (%)']:.2f}%")
    print("CA-RFA Model:")
    print(f"  Mean Accuracy: {summary['CA-RFA Model']['Mean Accuracy (%)']:.2f}%")
    print(f"  Standard Deviation: {summary['CA-RFA Model']['Std Deviation (%)']:.2f}%")
    
    return summary

# Example usage
if __name__ == "__main__":
    processed_folder = "./images_processed_typhoon"
    ca_folder = "./images_Reg_CA_model"
    ca_rfa_folder = "./images_Hybrid_Model"
    
    # Run batch comparison
    batch_comparison_results = batch_compare_models(processed_folder, ca_folder, ca_rfa_folder)