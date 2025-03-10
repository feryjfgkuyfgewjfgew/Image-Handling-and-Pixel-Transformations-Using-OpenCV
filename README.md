# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** NARESH.R
- **Register Number:** 21222340104

### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img =cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
#### 2. Print the image width, height & Channel.
```
image.shape
```
#### 3. Display the image using matplotlib imshow().
```
import cv2
import matplotlib.pyplot as plt
# Read the image in color mode
image = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)
# Convert BGR to RGB (Matplotlib uses RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.title("Eagle in Flight (RGB)")
plt.axis("off")  # Hide axes
plt.show()

```
#### 4. Save the image as a PNG file using OpenCV imwrite().
```
import cv2
# Read the image in color mode
image = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Save the image as a PNG file
    output_filename = 'Eagle_in_Flight.png'
    success = cv2.imwrite(output_filename, image)

    # Confirm if saving was successful
    if success:
        print(f"Image saved successfully as {output_filename}")
    else:
        print("Error: Failed to save the image.")

```
#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
import cv2
import matplotlib.pyplot as plt
# Read the saved PNG image (it may be grayscale)
image = cv2.imread('Eagle_in_Flight.png', cv2.IMREAD_UNCHANGED)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Convert grayscale to color (if necessary)
    if len(image.shape) == 2:  # Image is grayscale (2D shape)
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("Converted grayscale image to color (BGR).")
    else:
        image_color = image
        print("Image is already in color.")

    # Convert BGR to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.title("Converted Color Image")
    plt.axis("off")
    plt.show()

```
#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```
import cv2
import matplotlib.pyplot as plt
# Read the color image
image = cv2.imread('Eagle_in_Flight.png', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Get image dimensions
    height, width, channels = image.shape

    # Print width, height, and number of channels
    print(f"Image Width: {width} pixels")
    print(f"Image Height: {height} pixels")
    print(f"Number of Channels: {channels}")  # 3 for RGB/BGR

    # Convert BGR to RGB for correct color display in Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.title("Color Image")
    plt.axis("off")  # Hide axes
    plt.show()

```
#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```
import cv2
import matplotlib.pyplot as plt
# Read the color image
image = cv2.imread('Eagle_in_Flight.png', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define the cropping coordinates (adjust these manually)
    # Format: image[y_start:y_end, x_start:x_end]
    x_start, y_start = 100, 50  # Top-left corner
    x_end, y_end = 500, 400  # Bottom-right corner

    cropped_image = image[y_start:y_end, x_start:x_end]

    # Convert BGR to RGB for displaying in Matplotlib
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Display the cropped image
    plt.imshow(cropped_image_rgb)
    plt.title("Cropped Image - Eagle")
    plt.axis("off")
    plt.show()

    # Save the cropped image
    cv2.imwrite("Eagle_Cropped.png", cropped_image)
    print("Cropped image saved as 'Eagle_Cropped.png'")

```
#### 8. Resize the image up by a factor of 2x.
```
import cv2
import matplotlib.pyplot as plt
# Read the color image
image = cv2.imread('Eagle_in_Flight.png', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Get original dimensions
    height, width, channels = image.shape
    print(f"Original Dimensions: {width}x{height}")

    # Resize the image by a factor of 2x
    new_width = width * 2
    new_height = height * 2
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    print(f"Resized Dimensions: {new_width}x{new_height}")

    # Convert BGR to RGB for Matplotlib
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Display the resized image
    plt.imshow(resized_image_rgb)
    plt.title("Resized Image (2x)")
    plt.axis("off")
    plt.show()

    # Save the resized image
    cv2.imwrite("Eagle_Resized_2x.png", resized_image)
    print("Resized image saved as 'Eagle_Resized_2x.png'")

```
#### 9. Flip the cropped/resized image horizontally.
```
import cv2
import matplotlib.pyplot as plt
# Read the resized image (or use the cropped image)
image = cv2.imread('Eagle_Resized_2x.png', cv2.IMREAD_COLOR)  # Replace with 'Eagle_Cropped.png' if needed
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Flip the image horizontally (flipCode = 1)
    flipped_image = cv2.flip(image, 1)

    # Convert BGR to RGB for displaying with Matplotlib
    flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)

    # Display the flipped image
    plt.imshow(flipped_image_rgb)
    plt.title("Flipped Image (Horizontally)")
    plt.axis("off")
    plt.show()

    # Save the flipped image
    cv2.imwrite("Eagle_Flipped.png", flipped_image)
    print("Flipped image saved as 'Eagle_Flipped.png'")

```
#### 10. Read in the image ('Apollo-11-launch.jpg').
```
import cv2
import matplotlib.pyplot as plt
# Read the image in color mode
image = cv2.imread('Apollo-11-launch.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.title("Apollo 11 Launch")
    plt.axis("off")  # Hide axes
    plt.show()

```
#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```
import cv2
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Apollo-11-launch.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Get image dimensions
    height, width, _ = image.shape

    # Define text and font properties
    text = 'Apollo 11 Saturn V Launch, July 16, 1969'
    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2  # Adjust text size
    font_thickness = 2
    text_color = (255, 255, 255)  # White color
    text_background = (0, 0, 0)  # Black background

    # Get text size for positioning
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
    text_width, text_height = text_size

    # Define text position (centered at the bottom)
    x = (width - text_width) // 2
    y = height - 30  # 30 pixels from the bottom

    # Add a black rectangle background for better visibility
    padding = 10
    cv2.rectangle(image, (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + padding),
                  text_background, -1)

    # Put the text on the image
    cv2.putText(image, text, (x, y), font_face, font_scale, text_color, font_thickness)

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with text
    plt.imshow(image_rgb)
    plt.title("Apollo 11 Launch with Text")
    plt.axis("off")
    plt.show()

    # Save the modified image
    cv2.imwrite('Apollo-11-launch-with-text.jpg', image)
    print("Modified image saved as 'Apollo-11-launch-with-text.jpg'")

```
#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```
import cv2
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Apollo-11-launch.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define rectangle color (magenta) and thickness
    rect_color = (255, 0, 255)  # Magenta (BGR format)
    rect_thickness = 5  # Thickness of the rectangle border

    # Define the rectangle coordinates (manually adjust as needed)
    x_start, y_start = 250, 50   # Top-left corner of the rectangle
    x_end, y_end = 450, 900  # Bottom-right corner of the rectangle

    # Draw the rectangle
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), rect_color, rect_thickness)

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with the rectangle
    plt.imshow(image_rgb)
    plt.title("Apollo 11 Launch with Magenta Rectangle")
    plt.axis("off")  # Hide axes
    plt.show()

    # Save the modified image
    cv2.imwrite('Apollo-11-launch-with-rectangle.jpg', image)
    print("Modified image saved as 'Apollo-11-launch-with-rectangle.jpg'")

```
#### 13. Display the final annotated image.
```
import cv2
import matplotlib.pyplot as plt
# Read the final annotated image
image = cv2.imread('Apollo-11-launch-with-rectangle.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Final annotated image loaded successfully!")

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb)
    plt.title("Final Annotated Image - Apollo 11 Launch")
    plt.axis("off")  # Hide axes
    plt.show()

```
#### 14. Read the image ('Boy.jpg').
```
import cv2
import matplotlib.pyplot as plt
# Read the image in color mode
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.title("Boy Image")
    plt.axis("off")  # Hide axes
    plt.show()

```
#### 15. Adjust the brightness of the image.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image in color mode
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define brightness factor (increase or decrease)
    brightness_factor = 50  # Positive values increase brightness, negative decrease

    # Convert image to uint16 to prevent overflow during addition
    bright_image = np.clip(image.astype(np.int16) + brightness_factor, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for correct Matplotlib display
    bright_image_rgb = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)

    # Display the original and brightened images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Brightened image
    ax[1].imshow(bright_image_rgb)
    ax[1].set_title("Brightness Adjusted Image")
    ax[1].axis("off")

    plt.show()

    # Save the brightened image
    cv2.imwrite('Boy_brightness_adjusted.jpg', bright_image)
    print("Brightness adjusted image saved as 'Boy_brightness_adjusted.jpg'")

```
#### 16. Create brighter and darker images.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define brightness adjustment factors
    brightness_increase = 50  # Increase brightness
    brightness_decrease = -50  # Decrease brightness

    # Adjust brightness (increase and decrease)
    brighter_image = np.clip(image.astype(np.int16) + brightness_increase, 0, 255).astype(np.uint8)
    darker_image = np.clip(image.astype(np.int16) + brightness_decrease, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brighter_image_rgb = cv2.cvtColor(brighter_image, cv2.COLOR_BGR2RGB)
    darker_image_rgb = cv2.cvtColor(darker_image, cv2.COLOR_BGR2RGB)

    # Display Original, Brighter, and Darker images side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(brighter_image_rgb)
    ax[1].set_title("Brighter Image")
    ax[1].axis("off")

    ax[2].imshow(darker_image_rgb)
    ax[2].set_title("Darker Image")
    ax[2].axis("off")

    plt.show()

    # Save the modified images
    cv2.imwrite('Boy_brighter.jpg', brighter_image)
    cv2.imwrite('Boy_darker.jpg', darker_image)

    print("Brighter image saved as 'Boy_brighter.jpg'")
    print("Darker image saved as 'Boy_darker.jpg'")

```
#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define brightness adjustment factors
    brightness_increase = 50  # Increase brightness
    brightness_decrease = -50  # Decrease brightness

    # Adjust brightness (increase and decrease)
    brighter_image = np.clip(image.astype(np.int16) + brightness_increase, 0, 255).astype(np.uint8)
    darker_image = np.clip(image.astype(np.int16) + brightness_decrease, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brighter_image_rgb = cv2.cvtColor(brighter_image, cv2.COLOR_BGR2RGB)
    darker_image_rgb = cv2.cvtColor(darker_image, cv2.COLOR_BGR2RGB)

    # Display the Original, Darker, and Brighter images side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Darker image
    ax[1].imshow(darker_image_rgb)
    ax[1].set_title("Darker Image")
    ax[1].axis("off")

    # Brighter image
    ax[2].imshow(brighter_image_rgb)
    ax[2].set_title("Brighter Image")
    ax[2].axis("off")

    plt.show()

```
#### 18. Modify the image contrast.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define contrast factors
    contrast_factor1 = 1.1  # Slightly higher contrast
    contrast_factor2 = 1.2  # More contrast

    # Create transformation matrices
    matrix1 = np.array([[contrast_factor1, 0, 0], [0, contrast_factor1, 0], [0, 0, contrast_factor1]], dtype=np.float32)
    matrix2 = np.array([[contrast_factor2, 0, 0], [0, contrast_factor2, 0], [0, 0, contrast_factor2]], dtype=np.float32)

    # Apply contrast adjustment
    img_higher1 = np.clip(image.astype(np.float32) * contrast_factor1, 0, 255).astype(np.uint8)
    img_higher2 = np.clip(image.astype(np.float32) * contrast_factor2, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_higher1_rgb = cv2.cvtColor(img_higher1, cv2.COLOR_BGR2RGB)
    img_higher2_rgb = cv2.cvtColor(img_higher2, cv2.COLOR_BGR2RGB)

    # Display Original and Contrast Adjusted Images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Higher contrast (1.1x)
    ax[1].imshow(img_higher1_rgb)
    ax[1].set_title("Higher Contrast (1.1x)")
    ax[1].axis("off")

    # Higher contrast (1.2x)
    ax[2].imshow(img_higher2_rgb)
    ax[2].set_title("Higher Contrast (1.2x)")
    ax[2].axis("off")

    plt.show()

    # Save the contrast-adjusted images
    cv2.imwrite('Boy_higher_contrast_1.1.jpg', img_higher1)
    cv2.imwrite('Boy_higher_contrast_1.2.jpg', img_higher2)

    print("Contrast-adjusted images saved as 'Boy_higher_contrast_1.1.jpg' and 'Boy_higher_contrast_1.2.jpg'")

```
#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Define contrast factors
    lower_contrast_factor = 0.7  # Reduce contrast
    higher_contrast_factor = 1.3  # Increase contrast

    # Apply contrast adjustment
    lower_contrast_image = np.clip(image.astype(np.float32) * lower_contrast_factor, 0, 255).astype(np.uint8)
    higher_contrast_image = np.clip(image.astype(np.float32) * higher_contrast_factor, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_contrast_rgb = cv2.cvtColor(lower_contrast_image, cv2.COLOR_BGR2RGB)
    higher_contrast_rgb = cv2.cvtColor(higher_contrast_image, cv2.COLOR_BGR2RGB)

    # Display the Original, Lower Contrast, and Higher Contrast images side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Lower contrast image
    ax[1].imshow(lower_contrast_rgb)
    ax[1].set_title("Lower Contrast")
    ax[1].axis("off")

    # Higher contrast image
    ax[2].imshow(higher_contrast_rgb)
    ax[2].set_title("Higher Contrast")
    ax[2].axis("off")

    plt.show()

    # Save the contrast-adjusted images
    cv2.imwrite('Boy_lower_contrast.jpg', lower_contrast_image)
    cv2.imwrite('Boy_higher_contrast.jpg', higher_contrast_image)

    print("Contrast-adjusted images saved as 'Boy_lower_contrast.jpg' and 'Boy_higher_contrast.jpg'")

```
#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Split the channels
    B, G, R = cv2.split(image)

    # Create blank images for visualization
    blank = np.zeros_like(B)

    # Convert single channels to 3-channel images for proper visualization
    blue_channel = cv2.merge([B, blank, blank])  # Blue channel image
    green_channel = cv2.merge([blank, G, blank])  # Green channel image
    red_channel = cv2.merge([blank, blank, R])  # Red channel image

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blue_rgb = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2RGB)
    green_rgb = cv2.cvtColor(green_channel, cv2.COLOR_BGR2RGB)
    red_rgb = cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB)

    # Display Original and Color Channels
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Blue channel
    ax[1].imshow(blue_rgb)
    ax[1].set_title("Blue Channel")
    ax[1].axis("off")

    # Green channel
    ax[2].imshow(green_rgb)
    ax[2].set_title("Green Channel")
    ax[2].axis("off")

    # Red channel
    ax[3].imshow(red_rgb)
    ax[3].set_title("Red Channel")
    ax[3].axis("off")

    plt.show()

    # Save the channel images
    cv2.imwrite('Boy_Blue_Channel.jpg', blue_channel)
    cv2.imwrite('Boy_Green_Channel.jpg', green_channel)
    cv2.imwrite('Boy_Red_Channel.jpg', red_channel)

    print("Channel images saved as 'Boy_Blue_Channel.jpg', 'Boy_Green_Channel.jpg', and 'Boy_Red_Channel.jpg'")

```
#### 21. Merged the R, G, B , displays along with the original image
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Split the channels
    B, G, R = cv2.split(image)

    # Merge channels back to form the original image
    merged_image = cv2.merge([B, G, R])

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    merged_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

    # Display the Original and Merged images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Merged image
    ax[1].imshow(merged_rgb)
    ax[1].set_title("Merged Image (Reconstructed)")
    ax[1].axis("off")

    plt.show()

    # Save the merged image
    cv2.imwrite('Boy_Merged.jpg', merged_image)

    print("Merged image saved as 'Boy_Merged.jpg'")

```
#### 22. Split the image into the H, S, V components & Display the channels.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    H, S, V = cv2.split(hsv_image)

    # Display the Original and HSV Channels
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Original Image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Hue Channel
    ax[1].imshow(H, cmap='hsv')  # Hue uses HSV colormap
    ax[1].set_title("Hue Channel")
    ax[1].axis("off")

    # Saturation Channel
    ax[2].imshow(S, cmap='gray')  # Saturation is intensity-based
    ax[2].set_title("Saturation Channel")
    ax[2].axis("off")

    # Value Channel
    ax[3].imshow(V, cmap='gray')  # Brightness/Lightness
    ax[3].set_title("Value Channel")
    ax[3].axis("off")

    plt.show()

    # Save the HSV channel images
    cv2.imwrite('Boy_Hue_Channel.jpg', H)
    cv2.imwrite('Boy_Saturation_Channel.jpg', S)
    cv2.imwrite('Boy_Value_Channel.jpg', V)

    print("HSV channel images saved as 'Boy_Hue_Channel.jpg', 'Boy_Saturation_Channel.jpg', and 'Boy_Value_Channel.jpg'")

```
#### 23. Merged the H, S, V, displays along with original image.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('Boy.jpg', cv2.IMREAD_COLOR)
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully!")

    # Convert BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the image into H, S, V channels
    H, S, V = cv2.split(hsv_image)

    # Merge the channels back into an HSV image
    merged_hsv = cv2.merge([H, S, V])

    # Convert the merged HSV back to BGR for display
    merged_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

    # Convert BGR to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)

    # Display the Original and Merged Images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original Image
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Merged Image
    ax[1].imshow(merged_rgb)
    ax[1].set_title("Merged HSV Image")
    ax[1].axis("off")

    plt.show()

    # Save the merged image
    cv2.imwrite('Boy_Merged_HSV.jpg', merged_bgr)
    print("Merged image saved as 'Boy_Merged_HSV.jpg'")

```
## Output:
## **i)** Read and Display an Image.
## 1.Read 'Eagle_in_Flight.jpg' as grayscale and display: 
![image](https://github.com/user-attachments/assets/bcea0f9d-8ff8-4a5c-a470-625c79c22dad)
## 2.Save image as PNG and display:
![image](https://github.com/user-attachments/assets/90e33d36-b55f-4e59-9fa4-a9086863afd1)
## 3.Cropped image:
![image](https://github.com/user-attachments/assets/f723f5af-f3eb-4b0e-ab7d-a7f74dd69373)
## 4.Resize and flip Horizontally:
![image](https://github.com/user-attachments/assets/8c805c27-c060-465b-a09a-a2105d797138)
## 5.Read 'Apollo-11-launch.jpg' and Display the final annotated image:
![image](https://github.com/user-attachments/assets/5cadacd1-9926-4b2d-8839-881e756901e6)
## **ii)** Adjust Image Brightness. 
## 1.Create brighter and darker images and display:
![image](https://github.com/user-attachments/assets/cfc200dc-e391-435e-ba65-243d021801a9)
## **iii)** Modify Image Contrast. 
## Modify contrast using scaling factors 1.1 and 1.2:
![image](https://github.com/user-attachments/assets/e3ddd38a-06f2-4a57-b1c7-f8ac4d896e8e)
## **iv)** Generate Third Image Using Bitwise Operations.
## 1.Split 'Boy.jpg' into B, G, R components and display:
![image](https://github.com/user-attachments/assets/4b38148b-4eef-4290-8d66-1c84f8db8895)
## 2.Merge the R, G, B channels and display:
![image](https://github.com/user-attachments/assets/d0827c71-d204-4d40-a7fa-a2f3aebb9d82)
## 3.Split the image into H, S, V components and display:
![image](https://github.com/user-attachments/assets/70fdf513-f831-4d6b-aedc-4a0905a766f2)
## 4.Merge the H, S, V channels and display:
![image](https://github.com/user-attachments/assets/979b8a49-329b-44ed-a6dc-de67847fe25a)

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.
