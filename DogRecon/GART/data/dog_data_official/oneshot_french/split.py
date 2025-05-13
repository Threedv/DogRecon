from PIL import Image, ImageFilter

def split_image(image_path, output_path, size=(320, 320)):
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Calculate the number of rows and columns
    rows = height // size[1]
    cols = width // size[0]

    # Split the image into parts
    k = 0
    for row in range(rows):
        for col in range(cols):
            k +=1
            left = col * size[0]
            top = row * size[1]
            right = left + size[0]
            bottom = top + size[1]

            # Crop and save each part
            part = img.crop((left, top, right, bottom))
            resized_part = part.resize((512, 512), resample =Image.BICUBIC)
            resized_part.save(f"{output_directory}{k:04d}.png")

if __name__ == "__main__":
    # Specify your input and output paths
    input_image_path = "./images/output_0719_rgba.png"
    output_directory = "./images/"

    # Create the output directory if it doesn't exist
    import os
    os.makedirs(output_directory, exist_ok=True)

    # Split the image into parts
    split_image(input_image_path, output_directory)