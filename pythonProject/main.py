from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')  # ή 'Qt5Agg' ανάλογα με το ποιο backend λειτουργεί σωστά στο σύστημά σας
import matplotlib.pyplot as plt
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Φόρτωση της εικόνας
image_path = r'C:\Users\Vagelis\Desktop\pythonProject\cancer1.jpg'
image = Image.open(image_path)

# Επιλογή σημείων ελέγχου (σημεία στην εικόνα)
control_points = [(100, 50), (200, 100), (300, 200), (400, 150)]

# Υλοποίηση της Bézier καμπύλης
def de_casteljau(control_points, t):
    if len(control_points) == 1:
        return control_points[0]

    new_points = []
    for i in range(len(control_points) - 1):
        new_point = tuple(
            (1 - t) * control_points[i][j] + t * control_points[i + 1][j] for j in range(len(control_points[i])))
        new_points.append(new_point)

    return de_casteljau(new_points, t)

t_values = np.linspace(0, 1, 100)
curve_points = [de_casteljau(control_points, t) for t in t_values]

# Οπτικοποίηση της Bézier καμπύλης πάνω στην εικόνα
draw = ImageDraw.Draw(image)
draw.line(control_points, fill='red', width=2)
for point in control_points:
    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='red')

curve_x, curve_y = zip(*curve_points)
for x, y in zip(curve_x, curve_y):
    draw.point((x, y), fill='blue')

# Εμφάνιση της εικόνας με την Bézier καμπύλη σε ένα παράθυρο
fig1 = plt.figure()
fig1.canvas.manager.window.wm_geometry("+0+0")  # Set the window position to display on the primary screen

plt.imshow(np.array(image))
plt.title('Καμπύλη Bézier στην Εικόνα')

plt.show()

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PIL import Image
import numpy as np

# Initialize a counter for naming the saved files
save_counter = 0

# Function to handle the rectangle selection and save the cropped area
def onselect(eclick, erelease):
    global save_counter
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Convert the PIL image to a NumPy array
    im_np = np.array(im)

    # Crop the selected area
    im_crop = im_np[y1:y2, x1:x2]

    # Display the cropped area as an icon
    ax_crop.clear()
    ax_crop.imshow(im_crop)
    ax_crop.axis('off')
    canvas.draw()

    # Save the cropped area as an image file
    save_counter += 1
    save_path = f"cropped_image_{save_counter}.jpg"
    cropped_image = Image.fromarray(im_crop)
    cropped_image.save(save_path)
    print(f"Saved as {save_path}")

# Load the initial image
initial_image_path = r'C:\Users\Vagelis\Desktop\pythonProject\cancer1.jpg'
im = Image.open(initial_image_path)

# Create the main GUI window using tkinter
root = tk.Tk()
root.title("Image Cropping Tool")

# Create a Matplotlib figure and axis for the initial image
fig = Figure(figsize=(8, 6))
ax = fig.add_subplot(121)
ax.imshow(im)
ax.axis('off')

# Create a subplot for the cropped area (icon)
ax_crop = fig.add_subplot(122)  # Adjust the position and size as needed
ax_crop.axis('off')

# Create a RectangleSelector to select the area
rs = RectangleSelector(ax, onselect)

# Create a Matplotlib canvas for embedding the plot in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Start the tkinter main loop
root.mainloop()

# Φόρτωση της εικόνας
image_path = r'C:\Users\Vagelis\Desktop\pythonProject\cropped_image_1.jpg'
image = Image.open(image_path)

########################
# Επιλογή σημείων ελέγχου (σημεία στην εικόνα)
control_points = [(0, 0), (0, 90), (90, 90), (90, 0)]

# Υλοποίηση της Bézier καμπύλης
def de_casteljau(control_points, t):
    if len(control_points) == 1:
        return control_points[0]

    new_points = []
    for i in range(len(control_points) - 1):
        new_point = tuple(
            (1 - t) * control_points[i][j] + t * control_points[i + 1][j] for j in range(len(control_points[i])))
        new_points.append(new_point)

    return de_casteljau(new_points, t)

t_values = np.linspace(0, 1, 100)
curve_points = [de_casteljau(control_points, t) for t in t_values]

# Οπτικοποίηση της Bézier καμπύλης πάνω στην εικόνα
draw = ImageDraw.Draw(image)
draw.line(control_points, fill='red', width=2)
for point in control_points:
    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='red')

curve_x, curve_y = zip(*curve_points)
for x, y in zip(curve_x, curve_y):
    draw.point((x, y), fill='blue')

# Εμφάνιση της εικόνας με την Bézier καμπύλη σε ένα παράθυρο
fig1 = plt.figure()
fig1.canvas.manager.window.wm_geometry("+0+0")  # Set the window position to display on the primary screen

plt.imshow(np.array(image))
plt.title('Καμπύλη Bézier στην Εικόνα')

plt.show()
######################## Επιλογή περισσοτερων σημείων  ###############################################

# Επιλογή σημείων ελέγχου (σημεία στην εικόνα)
control_points = [(0, 0), (0, 90), (45, 45), (90, 90), (90, 0)]

# Υλοποίηση της Bézier καμπύλης
def de_casteljau(control_points, t):
    if len(control_points) == 1:
        return control_points[0]

    new_points = []
    for i in range(len(control_points) - 1):
        new_point = tuple(
            (1 - t) * control_points[i][j] + t * control_points[i + 1][j] for j in range(len(control_points[i])))
        new_points.append(new_point)

    return de_casteljau(new_points, t)

t_values = np.linspace(0, 1, 100)
curve_points = [de_casteljau(control_points, t) for t in t_values]

# Οπτικοποίηση της Bézier καμπύλης πάνω στην εικόνα
draw = ImageDraw.Draw(image)
draw.line(control_points, fill='red', width=2)
for point in control_points:
    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='red')

curve_x, curve_y = zip(*curve_points)
for x, y in zip(curve_x, curve_y):
    draw.point((x, y), fill='blue')

# Εμφάνιση της εικόνας με την Bézier καμπύλη σε ένα παράθυρο
fig1 = plt.figure()
fig1.canvas.manager.window.wm_geometry("+0+0")  # Set the window position to display on the primary screen

plt.imshow(np.array(image))
plt.title('Καμπύλη Bézier στην Εικόνα')

plt.show()
######################## Επιλογή περισσοτερων σημείων  ###############################################

# Επιλογή σημείων ελέγχου (σημεία στην εικόνα)
control_points = [(0, 0), (30, 30), (45, 45), (60, 60), (90, 90),(100,100)]

# Υλοποίηση της Bézier καμπύλης
def de_casteljau(control_points, t):
    if len(control_points) == 1:
        return control_points[0]

    new_points = []
    for i in range(len(control_points) - 1):
        new_point = tuple(
            (1 - t) * control_points[i][j] + t * control_points[i + 1][j] for j in range(len(control_points[i])))
        new_points.append(new_point)

    return de_casteljau(new_points, t)

t_values = np.linspace(0, 1, 100)
curve_points = [de_casteljau(control_points, t) for t in t_values]

# Οπτικοποίηση της Bézier καμπύλης πάνω στην εικόνα
draw = ImageDraw.Draw(image)
draw.line(control_points, fill='red', width=2)
for point in control_points:
    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='red')

curve_x, curve_y = zip(*curve_points)
for x, y in zip(curve_x, curve_y):
    draw.point((x, y), fill='blue')

# Εμφάνιση της εικόνας με την Bézier καμπύλη σε ένα παράθυρο
fig1 = plt.figure()
fig1.canvas.manager.window.wm_geometry("+0+0")  # Set the window position to display on the primary screen

plt.imshow(np.array(image))
plt.title('Καμπύλη Bézier στην Εικόνα')

plt.show()

##################### Εντοπισμος Καρκινου με την χρήση έτοιμου μοντέλου ##############################

import tensorflow as tf
import tensorflow_hub as hub

# Create a Sequential model with the MobileNet V2 KerasLayer
m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5")
])

# Build the model with the desired input shape
m.build([None, 224, 224, 3])  # Batch input shape.

# Load an image that you want to classify
#image_path = "path/to/your/image.jpg"
image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
image = tf.keras.utils.img_to_array(image)
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# Preprocess the image (normalize it)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

# Make predictions
predictions = m.predict(image)

# Decode the predictions (assuming you want class labels)
imagenet_labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://download.tensorflow.org/data/ImageNetLabels.txt")
imagenet_labels = []
with open(imagenet_labels_path) as f:
    imagenet_labels = f.read().splitlines()

predicted_class_index = tf.argmax(predictions, axis=1)
predicted_class_label = imagenet_labels[predicted_class_index[0]]

# Display the image and label
plt.figure(figsize=(8, 8))
plt.imshow(image[0])
plt.axis('off')
plt.title(f"Predicted class: {predicted_class_label}")
plt.show()