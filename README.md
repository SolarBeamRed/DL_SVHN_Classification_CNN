# Handmade CNN for SVHN Digit Classification (PyTorch)
#### Modular Deep Learning project using Pytorch for classification of Street View House Numbers dataset. Built using custom CNN. Contains training scripts, notebooks, figures, reports and final model's parameters<br>Accuracy achieved: 0.9543

___
### TLDR
Chose to build a custom CNN model instead of tuning pretrained ones. Chose the SVHN dataset to work on for the project.
No rigorous EDA was necessary, since dataset contained only images. Used data augmentation during training. Tuned hyperparameters a lot using Optuna. 
Spent a few hours training tuned model. Then saved model and checkpoints, wrote training scripts and organised remaining project structure.

___

## Objective
- Build a CNN from scratch for multi-class image classification  
- Face, understand and debug common deep learning pipeline issues  
- Implement a stable and correct training + validation pipeline  
- Perform extensive tuning of hyperparameters using Optuna  
- Improve model performance through architecture and data pipeline refinement  

___

## Dataset
**SVHN (Street View House Numbers)**  
- Real-world digit classification dataset (0–9)  
- Images of size 32×32 (resized to 28×28 during preprocessing)  

Dataset source: http://ufldl.stanford.edu/housenumbers/  

Note: Dataset is **not included** in the repository.  
If using torchvision.datasets.SVHN, then provide root='data' when creating instance of SHVN.
If using provided link, place `.mat` files inside the `data/` directory.

___

## Approach

### General Approach
1. Build a baseline CNN and ensure correct training behavior  
2. Debug issues related to shapes, channels, and loss computation  
3. Implement proper train/validation split and data augmentation for train split
4. Tune hyperparameters using Optuna 
5. Implement better architecture (BatchNorm, Dropout, Adaptive Pooling) and optimizers obtained through tuning
6. Final training with early stopping and checkpointing  

---

### Data Handling
- Used **train split** for training + validation  
- Used **extra split** to increase training data  
- Validation set created from train split (no augmentation)  
- Proper transform separation:
  - Train:  augmentation + normalization  
  - Validation/Test: normalization only  

---

### Data Augmentation
Used albumentations library for augmentation. Used augmentations include:
- Random cropping with padding  
- Color jitter (brightness, contrast, saturation)
- Very slight blur and rotations
- No heavy rotations to preserve digit structure  

---

### Model
Custom CNN with:

- Conv → ReLU → MaxPool → BatchNorm blocks  
- Progressive channel increase  
- MaxPooling for spatial reduction  
- AdaptiveAvgPool2d to remove dependency on input size  
- Fully connected classifier with dropout  

---

### Training
- Loss: CrossEntropyLoss  
- Optimizer: Adam (tuned parameters)  
- Early stopping based on validation loss  
- Model checkpointing (best model saved)  

---

### Hyperparameter Tuning
- Used **Optuna** with SQLite storage  
- Tuned:
  - Learning rate  
  - Weight decay
  - Optimizer choice and related beta parameters
  - Dropout  
  - Architecture-related parameters, such as number of Conv and dense layers, and number of filters/neurons for each.
  - Usage of BatchNorm for dense layers (better results without BatchNorm for dense layer) 

---

## Results

Final tuned CNN:

- **Test Accuracy: ~95.43%**
- Stable training and validation curves (figures available in reports/figures) 
- No significant overfitting observed  

---

## Key Insights

- Saving checkpoints is incredibly useful, especially to train across multiple sessions
- Tuning was very helpful in improving accuracy beyond 0.90  
- AdaptiveAvgPool2d simplified architecture and improved generalization
- Data augmentation improved robustness across epochs
- Saving Optuna study results is helpful for persistence and analysis

---

## Project Structure
src/ # Core training pipeline<br>
│<br>
├── models/ &emsp;# CNN architecture<br>
├── data/ &emsp;&emsp;&nbsp;# Data loading and transforms<br>
├── utils/ &emsp;&emsp;&nbsp;&nbsp;# Configs and helpers<br>

notebooks/ &emsp;&emsp;# Experimentation and tuning<br>
reports/ &emsp;&emsp;&emsp;&ensp;# Plots and Optuna outputs<br>
configs/ &emsp;&emsp;&emsp;&ensp;# config.json<br>
checkpoints/ &emsp;&nbsp;# saved models and checkpoints<br>
data/ &emsp;&emsp;&emsp;&emsp;&ensp;# SVHN .mat files (not included)<br>

___

## How to Run

1. Download SVHN dataset:<br>
   - Use torchvision.datasets.SVHN with root='data'<br>
   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;OR
   - Download from http://ufldl.stanford.edu/housenumbers/ and place `.mat` files inside `data/`  

<br>
2. Install dependencies: <br>

~~~
pip install -r requirements.txt
~~~

3. Train model: <br>
~~~
python src/train.py
~~~

4. Evaluate model:<br>
~~~
python src/evaluate.py
~~~

---

## Notes

- Dataset is not included due to size  
- Model is trained from scratch (no pretrained networks used)  
- Focus is on understanding and building the pipeline  

---

## Reflections

Was a pretty fun project ngl. Performance would obviously be better if pretrained models
were used, but I just wanted to handcraft a CNN myself. Augmentations with albumentations
library was fun too. Tuning felt exciting as well, cause you're curious on how your new parameters will perform.
Sometimes it's a pleasant improvement, other times, it's a disappointment, but overall,
still exciting. Was pretty fun seeing the changes as you tweaked the architecture in real time. Overall, fun little project