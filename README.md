# pix2pix-edges-to-images

## Demo

## How to train a pix2pix(edges2fish) model from scratch
- prepare data
- Detect edges of all images
- Images combination

## Prepare Data

### prepare images using Photoshop
I use my drawing to created my own fish datase https://www.instagram.com/fishchief/?hl=zh-tw. Most of my previous works are well-organised, I import them into photoshop to do some basic adjustment such as remove the background, manually enhance edges, etc.
<img src="images/AdjustImage.jpg" alt="alt text" width="500">

### Save each layers from PSD file
<img src="images/148Layers.png" alt="alt text" width="500">
I imput all of the images into the same file, after adjusted all of them, I uploaded this psd file to Google Drive, running the code below can export every layers to target folder also on Google Drive.

``` 
# export layers from psd
from psd_tools import PSDImage
from PIL import Image

# path to PSD file
psd = PSDImage.open('/content/drive/MyDrive/fishDataset/2306131706.psd')

# choose the output folder
output_folder = '/content/drive/MyDrive/fishDataset/png_folder'
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

### Resize PNG images and save them as jpg into resized folder

### Detect edges of all images and combination
I upload all the images to my google drive, after mounting the colad with my drive, I started to detect edges of my images then combine them together. The script that I use to detect edges of images and for combination from one folder at once is here:
https://github.com/yining1023/pix2pix-tensorflow/blob/master/tools/edge-detection.py
(need to specify our own path on line 31), also need to create a new empty folder in dataset folder called "edges" in the same directory.

### Images combination
```
number_of_file = 67
folder = '/content/drive/MyDrive/fishDataset/combined'

for i in range(number_of_file):
  edges = np.array(Image.open(f'/content/drive/MyDrive/fishDataset/edges/{i+1}.jpg'))
  resized = np.array(Image.open(f'/content/drive/MyDrive/fishDataset/resized/{i+1}.jpg'))
  # (256,256,1)
  edges = tf.expand_dims(edges,2)
  edges = tf.concat([edges,edges,edges],2)
  img_combined = tf.concat([resized,edges], 1)
  img_pil = tf.keras.utils.array_to_img(img_combined)
  img_pil.save(f'{folder}/{i}.jpg')
```
This code snippet was written by Jasper, aime to combine input image and target image together

## Model Training

Used the code in this link https://gitlab.cern.ch/smaddrel/pix2pix-tf_2_0/-/blob/master/pix2pix.py to train the model.
If you're using dataset fron online, paste the URL to line 16 and unzip it with line 18-20. But since I'm using my own dataset, I commanded both part of code and add my root path which is in my drive on line 22 (Don't forget to mount Drive!).

Then I create 2 sub folders in base folder, ```train``` and ```val```, and manually split images in dataset into these two folders.

To test if the model work, I changed the EPOCHS number from 200 into 3 in line 477, after running the code, it will automatically create a couple new sub-folders in content(content/results/0). By running following code:
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
(code was provided by ChatGPT), here's the result of epoch = 3<br>
<img src="images/resultOfEPOCH=3.png">

Then I found I shouldn't use green or black in my target image unleast I want the background of predict images are not white.

## Result

## Export the model

## Getting Start
