# Overview
This project is a response to a test assignment from datel to assess the competence of candidates for ML Expert.

This test requires me to develop a semantic segmentation model that is capable of identifying pixel regions of buildings from satellite images. Specifically, I need to process and model a Python pickle file containing satellite images and corresponding mask data. The task can be divided into several key steps involving data processing, model training, evaluation and packaging. 

# Introduction to the dataset
- Data file: dataset.pickle.
- Data structure: the file contains the training set and validation set in the format of a list, where each element of the list is a dictionary. Each dictionary has two keys:
  - image: stores the satellite image which has three channels corresponding to sensor readings in the blue, green, and red bands.
  - mask: This is the corresponding mask data that indicates which pixels in the image contain buildings and which do not. The mask is a binary matrix, 1 for buildings and 0 for background.

# Project Objectives
- Train a semantic segmentation model
- Evaluate model performance
- Model Packaging and Inference
  
# Technical Methodology
- Model building & training: PyTorch, UNet, Dice Loss, Adam
  - Dataset Preprocessing: A custom dataset class SatelliteDataset is defined, inheriting from PyTorch's Dataset class. This class encapsulates the satellite image data and corresponding masks.

  - UNet Model Architecture: the goal of this model is to segment images by predicting a binary mask, where each pixel is classified as belonging to a foreground object or the background. In this case, the model takes an input image of size (3, H, W) (with 3 channels for RGB images), and outputs a binary mask of size (1, H, W) where each pixel has a probability value between 0 and 1, representing the likelihood that the pixel belongs to the foreground object.

Encoder (Downsampling Path): The encoder progressively reduces the spatial dimensions of the input while increasing the number of feature channels. It includes:

Two convolutional layers followed by ReLU activations.

Dropout layers after each ReLU activation to regularize the model by dropping a fraction of the neurons during training.
MaxPooling layer to reduce the spatial dimensions by half, allowing the model to capture more abstract features at lower resolutions.

Decoder (Upsampling Path): The decoder path mirrors the encoder, but instead of downsampling, it upscales the feature maps back to the original image size. It includes:

Transposed convolution (also known as deconvolution) for upsampling the feature maps, increasing their spatial dimensions.
ReLU activations followed by Dropout to regularize the upsampling process.

A final convolution layer with a Sigmoid activation, which outputs a single-channel probability mask, where each pixel represents the likelihood of being foreground.

  - Loss Functions: tried Dice Loss, Focal Loss and Combined Loss. 

  - Training: 
In each epoch, the model performs forward passes on the input data, computes the loss, backpropagates the gradients, and updates the model weights using the Adam optimizer. 

The model is trained for a specified number of epochs, and the average loss per epoch is printed for monitoring training progress.

- Evaluation Metrics: IoU, Accuracy, Precision, Recall, F1-score
  - After training, the model is evaluated using a variety of metrics such as IoU, Dice Coefficient, Precision, Recall, and F1 Score.

  - The evaluate_model function performs validation at different probability thresholds (e.g., 0.1, 0.3, 0.5, 0.6, 0.7) and calculates the segmentation metrics.

- Packaging and Publishing: notebook, script, command to run script(conda environment)
  - After training, the trained model is saved to a file for later inference.

  - The run_inference function loads this pre-trained model and uses it for inference on the validation dataset, then evaluates the model's performance using the same metrics as during validation.


# Final Results
- A working semantic segmentation model
  - See the building process in the dateltest.ipynb
  - See the saved model in unet_model.pth

- An evaluation report on a validation set(take one threshold(0.3) as an example ) 
  - Validation threshold: 0.3
  - Validation IoU: 0.1124
  - Validation Dice Coefficient: 0.1964
  - Validation Precision: 0.1142
  - Validation Recall: 0.8684
  - Validation F1 Score: 0.1964

IoU measures the overlap between the predicted mask and the true mask. The value of 0.1124 means the overlap between the predicted mask and the true mask is very low, suggesting that the model is not performing well in identifying the correct regions.

Similar to IoU, the Dice Coefficient measures overlap, but it is a bit more sensitive to smaller object regions. A value of 0.1964 indicates low overlap between the predicted and true masks, similar to the IoU result.

Precision is the proportion of correctly predicted positive pixels (true positives) out of all pixels predicted as positive. A value of 0.1142 suggests that only around 11.42% of the pixels predicted as "foreground" (buildings) by the model were actually correct. This means the model is predicting a lot of false positives (incorrectly classifying background as buildings).

Recall measures the proportion of actual positive pixels that the model correctly identified. The very high value of 0.8684 means that the model is predicting the majority of  actual positives but is likely over-predicting, as seen by the very low precision.

The F1 Score is the harmonic mean of precision and recall. It balances the two metrics, providing a single measure of the model's accuracy. A value of 0.1964 is low, reflecting the poor balance between precision and recall. While recall is very high, the low precision drags down the F1 score.

- A packaged tool that allows the user to run the inference and view the results
  - In order to run the notebook and script successfully in a conda environment, you need to install necessary dependencies: torch, numpy, matplotlib, import_ipynb. Refer to requirement.txt.

  - See the command-line application in inference.py.

  - Command to run this script: python inference.py --model path_of_model --dataset path_of_data; for example, if you put the model and validation data in the same file as notebook and script, and name the model like the one below, then: python inference.py --model unet_model.pth --dataset dataset.pickle



# Ideas for Further Improvements

- Class Balance Techniques:
  - Apply data augmentation to increase the diversity of the training set.
  - Use class weighting in the loss function to penalize incorrect predictions more heavily.
  - Explore Oversampling or Undersampling to balance the number of samples in each class.

- Tune Hyperparameters:
  - Adjust the weights of the Combined Loss.
  - Experiment with different learning rates and optimizers to improve convergence.

- Use a Validation Split in Training:
  - Introduce a validation split in the training process to monitor the model's performance during training and avoid overfitting. 


