# pix2pix-edges-to-images

## Demo
<img src="images/Demo.jpg" alt="alt text" width="1000">
Video Link: https://www.youtube.com/watch?v=GNRUXdwpvnY&ab_channel=hanhsunshih 

## How to train a pix2pix(edges2fish) model from scratch
- prepare data<br>
  └ prepare images using Photoshop<br>
  └ Save each layer(.png) from PSD file by Python<br>
  └ Resize PNG images and save them as jpg into resized folder<br>
  └ Detect edges of all images<br>
  └ combine target(resized) image and edge image<br>
- Model Training

## Prepare Data

### prepare images using Photoshop
<img src="images/prepareImagesUsingPhotoshop.jpg" alt="alt text" width="800">
I use my drawing to created my own fish datase https://www.instagram.com/fishchief/?hl=zh-tw. Most of my previous works are well-organised, I import them into photoshop to do some basic adjustment such as remove the background, manually enhance edges, etc.<br>
<img src="images/AdjustImage2.jpg" alt="alt text" width="800">

### Save each layer from PSD file by Python
<img src="images/148Layers.jpg" alt="alt text" width="800">
I gathered all of the images into the same file, there're a lot of layers, if I need to manually save them one by one will take too many times. So I uploaded the psd file to Google Drive, running the code below to export every layers to target folder(png_folder).

```python
# export layers from psd
from psd_tools import PSDImage
from PIL import Image

# path to PSD file
psd = PSDImage.open('/content/drive/MyDrive/fishDataset/2306131706.psd')

# choose the output folder
output_folder = '/content/drive/MyDrive/fishDataset/'
os.makedirs(output_folder, exist_ok=True)

for index, layer in enumerate(psd):
    # use layer's name as file name
    layer_name = f'layer_{index}.png'
    output_path = os.path.join(output_folder, layer_name)

    # save as PNG file
    layer_image = layer.topil()
    layer_image.save(output_path, 'PNG')

    print(f'Saved layer {layer_name}')
``` 
(code provided by ChatGPT)

### Resize PNG images and save them as jpg into resized folder
<img src="images/resize.jpg" alt="alt text" width="800">
Size of PNG files which has been exported will suit the shape of the layer. The training model I'm going to use only accept image which is 256*256px, so I need to resize them and also save them as jpg file so they don't have alpha channel. (Not sure if the model allow me to use PNG file🤔) JPG files will be exported to the file called "resized".
The directory structure is organized as follows:

```
Mydrive/
├── fishDataset/
│   └── file.psd/
│   └── png_folder/
│   └── resized_folder/
│   └── edges_folder/
│   └── combined_folder/
│       └── train_folder/
│       └── val_folder/
```

```python
import os
from PIL import Image

number_of_png = 122
png_folder = "/content/drive/MyDrive/fishDataset/png_folder"
output_folder = "/content/drive/MyDrive/fishDataset/resized"
target_size = (256, 256)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    max_size = max(width, height)
    new_img = Image.new(pil_img.mode, (max_size, max_size), background_color)
    new_img.paste(pil_img, ((max_size - width) // 2, (max_size - height) // 2))
    return new_img

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for i in range(number_of_png):
    # Load the PNG image
    png_path = os.path.join(png_folder, f"layer_{i+1}.png")
    png = Image.open(png_path)
    #print()

    # Resize the image to the target size
    resized = expand2square(png, (255, 255, 255, 255)).resize(target_size)
    #resized = expand2square(png.resize(target_size), (255, 192, 203, 255))
    display(resized)

     # Create a new image with white background
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    
    # Composite the resized image onto the new image while preserving transparency
    new_img.paste(resized, (0, 0), mask=resized.split()[3])

    # Convert and save as JPG
    jpg_path = os.path.join(output_folder, f"{i+1}.jpg")
    new_img.save(jpg_path)

    print(f"Processed image {i+1}/{number_of_png}")

print("Resizing complete!")
```
Reference: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/

### Detect edges of all images
<img src="images/detectEdge.jpg" alt="alt text" width="800">
After saving all the resized images to my resized_folder, I started to detect edges of my images and save the edge photo into edges_folder. The script I used to detect edges of images is here:
https://github.com/yining1023/pix2pix-tensorflow/blob/master/tools/edge-detection.py<br>
(need to specify our own path on line 31)

### Combine target(resized) image and edge image
This step I'm going to combine resized image and edge image together, so I can save them into combined_folder and split them into train_folder and val_folder.
<img src="images/combination.jpg" alt="alt text" width="800">

``` python
number_of_file = 122
folder = '/content/drive/MyDrive/fishDataset/combined'

for i in range(number_of_file):
  edges = np.array(Image.open(f'/content/drive/MyDrive/fishDataset/edges/{i+1}.jpg'))
  resized = np.array(Image.open(f'/content/drive/MyDrive/fishDataset/resized/{i+1}.jpg'))
  #(256,256,1)
  edges = tf.expand_dims(edges,2)
  edges = tf.concat([edges,edges,edges],2)
  img_combined = tf.concat([resized,edges], 1)
  img_pil = tf.keras.utils.array_to_img(img_combined)
  img_pil.save(f'{folder}/{i}.jpg')
```
This code snippet was written by Jasper, aime to combine input image and target image together

### Randomly split combined image into train/ val folder
80% of data goes to train_folder, 20% goes to val_folder

```python
import os
import random
import shutil

source_folder = "/content/drive/MyDrive/fishDataset/combined"  # Path to the source folder containing images
destination_folder_b = "/content/drive/MyDrive/fishDataset/combined/train"  # Path to the destination folder B
destination_folder_c = "/content/drive/MyDrive/fishDataset/combined/val"  # Path to the destination folder C
split_ratio = 0.8  # 80% for folder B, 20% for folder C

# Get the list of image files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Shuffle the image files randomly
random.shuffle(image_files)

# Split the image files based on the split ratio
split_index = int(len(image_files) * split_ratio)
files_for_folder_b = image_files[:split_index]
files_for_folder_c = image_files[split_index:]

# Move files to destination folder B
for file_name in files_for_folder_b:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder_b, file_name)
    shutil.move(source_path, destination_path)

# Move files to destination folder C
for file_name in files_for_folder_c:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder_c, file_name)
    shutil.move(source_path, destination_path)
```

## Model Training

I Used the code in this link https://gitlab.cern.ch/smaddrel/pix2pix-tf_2_0/-/blob/master/pix2pix.py to train the model.
If you're using dataset fron online, paste the URL to line 16 and unzip it with line 18-20. But since I'm using my own dataset, I commanded both part of code and add my root path from Google Drive on line 22.

Following is the structure of the code, I found it more understandable to read this then directly try to understand all of the code.
<img src="images/trainingCodeStructure.png" alt="alt text" width="600">

To test if the model work, I changed the EPOCHS number from 200 to 3 in line 477, after running the code, it will automatically create a couple of new sub-folders (content/results/0). By running following code:
```
import matplotlib.pyplot as plt
import os

folder_path = '/content/results/4'

file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]

for file_path in file_paths:
    image = plt.imread(file_path)
    plt.imshow(image)
    plt.show()
```
(code provided by ChatGPT), here's the result of epoch = 3, data size: 22<br>
<img src="images/resultOfEPOCH=3.png">



## Result
<img src="images/result1.jpg" alt="alt text" width="1000">
<img src="images/result2.jpg" alt="alt text" width="1000">

I tried to adjust different parameters to see how can I get better result, including number of EPOCHS, counter (iteration limit), size of dataset. After several times of testing, I found EPOCHS is the key influence of the result. As we set EPOCHS which is more than 100, the result will be much better. But in order to be concern of the training time, decrease the counter from 100 to 25 might take fewer time for training.

## What else can I improve the result?
Since my dataset is quite small (only has 149 imges), in the future, I can do data augmentation such as flipping, rotate to increase the dataset. Using other model came after pix2pix, for example, pix2pixHD might also be an effective way to improve the result.
