# Manipulating Deep Neural Network Reveals Optimal Regularization in Angry Face Coding and Its Dysregulation in Depression

Step 1 Computation of visual emotion code
-------
### 1. Train the VAE model
* Use [imagen_main_cv.py](https://github.com/hanluyt/EmotionConcept_reg/blob/main/code_VAE/imagen_main_cv.py)  to perform hyperparameter selection for the VAE model.
* Hyperparameters are determined using 5-fold cross-validation.
  
### 2. Compute the visual emotion code
* Once the optimal hyperparameters are selected, use [total_main.py](https://github.com/hanluyt/EmotionConcept_reg/blob/main/code_VAE/total_main.py) to train the final VAE model with data from all participants at age 19.

Note: You can modify the `directory` and `save_dir` in the [vae.yaml](https://github.com/hanluyt/EmotionConcept_reg/blob/main/code_VAE/vae.yaml) to suit your setup.

