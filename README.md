# Text2Img-GAN
Created a text to image generator from using the CIFAR10 dataset 

My process works by taking embeddings from CLIP, and adding with a noise vector to input into the generator. 

The training loss for the discriminator is a BCE between the real and predicted labels and for the generator I use the adversarial loss with feature matching loss and similarity loss. 

The process was additionally optimized with a gradient penalty on the discriminator and label smoothing in order to obtain better results. 

I also have a script that outputs generated images from text using StableDiffusion in this, which can be run by "python3 test.py" after going into that folder

# Rendering images with my weights

First, download all the files. 

Currently, the model only supports certain inputs, such as:
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'

If you would like to test the model out, run the code "python3 render.py" to render the image, which will output in the rendered_images folder. 

# Training with your own data

1. Place your data into a Data/ folder and update the img and annotations directories in the main.py file. 
2. Create a dataset using the data that you downloaded and the transforms in the file
3. run "python3 main.py"

NOTE: this code uses the mac GPU mps, so if you would like to use a different GPU, make sure to edit those if statements setting the device accordingly. 
