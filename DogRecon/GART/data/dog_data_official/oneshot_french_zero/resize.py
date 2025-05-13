from PIL import Image
import os
# Open an image file

from PIL import Image
import os

# Set the path to the folder containing your images
folder_path = "./images"

# Set the desired size for the resized images
new_size = (512, 512)  # Replace width and height with your desired dimensions

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Add more file extensions if needed
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)

        # Open the image
        original_image = Image.open(image_path)

        # Resize the image
        resized_image = original_image.resize(new_size)

        # Save the resized image (optional, you can skip this if you don't want to save)
        resized_image.save(os.path.join(folder_path,filename))