from PIL import Image

def change_colors(input_image_path, output_image_path):
    # Open the image
    img = Image.open(input_image_path)
    img = img.convert("RGBA")  # Ensure image is in RGBA mode
    
    # Resize to 462 x 462 pixels
    img = img.resize((462, 462))

    # Get image data
    pixels = img.load()
    
    # Loop through each pixel
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b, a = pixels[x, y]
            if a > 0:  # Ignore fully transparent pixels
                # Replace red with gray
                if r > g and r > b:
                    gray = int((r + g + b) / 3)
                    pixels[x, y] = (gray, gray, gray, a)
                # Replace blue with another color
                elif b > r and b > g:
                    pixels[x, y] = (0, 0, 255, 255)  # Example: orange color
            else:
                # Change the background to blue (for transparent pixels)
                pixels[x, y] = (0, 0, 255, 255)  # Solid blue

    # Add an extra loop to ensure background areas are blue
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b, a = pixels[x, y]
            if (r, g, b) != (0, 0, 255, 255) and r != g:  # If not orange or gray
                pixels[x, y] = (0, 0, 255, 255)  # Change to blue

    # Save the modified image
    img.save(output_image_path)
    print(f"Modified and resized image saved to {output_image_path}")

# Example usage
input_image = r"images_trash_pile\trash4.png"  # Replace with your image file path
output_image = r"images_processed_trash\processedTrashImg.png"
change_colors(input_image, output_image)
