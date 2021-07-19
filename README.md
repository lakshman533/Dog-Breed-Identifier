# Dog-Breed-Identifier

Dogs are the most variable mammal on earth, with artificial selection producing around 450 globally recognized dog breeds. These breeds possess distinct traits related to morphology, which include body size, skull shape, tail phenotype, fur type, and coat color. Their behavioral traits include guarding, herding, and hunting, and personality traits such as hyper social behavior, boldness, and aggression. Most breeds were derived from small numbers of founders within the last 200 years. As a result, today dogs are the most abundant carnivore species and are dispersed around the world.

<p align="center">
     <img width="500" src="https://assets.rbl.ms/10891416/980x.gif" alt="Group of Dogs">
</p>

This is where we get a chance to apply Data Science and Machine Learning techniques to enable computers to fill the gap that human beings struggle to consistently fill.

# Objective

The project is part of the Udacity’s Data Scientist nanodegree program and the goal is to use a Convolutional Neural Network (CNN) to classify images of dog’s according to their breed when given a picture of the dog as input.

We did the following in our project:

1. Detect human faces in the input image
2. Detect dog faces in the input image
3. Predict the dog’s breed if it detects a dog
4. Build a CNN using transfer learning to Classify Dog Breeds
5. Write an algorithm and test it.

# Libraries

Libraries that are used in out code are :

* Image processing - OpenCV (cv2), PIL

* Keras for creating CNN

* matplotlib for viewing plots/images and numpy for tensor processing

* Utility libraries - random (for random seeding), glob(for folder and path operations), tqdm (for execution progress), sklearn (for loading datasets)

# Detect human faces in the input image

OpenCV’s implementation of Haar feature-based cascade classifiers is used to detect human faces in this project. OpenCV provides many pre-trained face detectors, stored as XML files on GitHub. We have downloaded one of these detectors and stored it in the haarcascades directory.

# Detect dog faces in the input image

In this section, we use a pre-trained ResNet-50 model to detect dogs in images.

The process of detecting the dogs in the input images consisted of two steps. The first step was to process the incoming image data and the second step was making predictions based on the processed images.

# Predict the dog’s breed if it detects a dog

Even though the goal of the project is to build a dog breed classifier using transfer learning, this part of the project was useful for understanding the fundamental of CNNs and how they work.

The target for this task was to build a CNN with an accuracy greater than >1%. The network described above achieved 7.665% without any fine-tuning of parameters or data augmentation.

# Build a CNN using transfer learning to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we can train a CNN using transfer learning. In the following step, we used transfer learning to train your own CNN.
I chose to build my CNN by leveraging a Resnet 50 CNN trained on the ImageNet database.

The accuracy which I obtained is 78.3% which is a good accuracy score.

# Write an Algorithm and test it.

Here we are creating our algorithm to analyze any image. The algorithm accepts a file path and:

if a dog is detected in the image, return the predicted breed.

if a human is detected in the image, return the resembling dog breed.

if neither is detected in the image, provide output that indicates an error.

And I have tested the algorithm and the results are far better than what I expected.

# Results

The output is better than I expected. Even If there is a human face in the picture, the algorithm matches it very well. However for very close-looking dog races, it still seems to be tricky to get it right.

So to improve our Algorithm, we can do the following

Increase the training data by including more pictures of different dog breeds.

The main findings of the code can be found at the post available [here](https://lakshmanraj23.medium.com/identifying-dog-breeds-using-convolutional-neural-networks-e1a039db87e4)

Image Augmentation

Accuracy can also be improved by increasing the depth of the neural network.

# Licensing, Authors, Acknowledgements

Special thanks to @udacity for providing this wonderful project and various kaggle and github kernels.
