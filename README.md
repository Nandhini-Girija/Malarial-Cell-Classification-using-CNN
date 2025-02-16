# Malarial Cell Classification Using CNN  

## **Project Overview**  
This project aims to classify **red blood cells** as **parasitized (infected) or uninfected** using **Convolutional Neural Networks (CNNs)**. Malaria is a life-threatening disease, and early detection is crucial for effective treatment. This **deep learning-based approach** automates and improves the accuracy of malaria diagnosis.  

## **Features**  
- **Dataset Preprocessing**: Loads and normalizes images of infected and uninfected cells.  
- **CNN-Based Classification**: A deep learning model with three convolutional layers for feature extraction.  
- **Model Training & Evaluation**: Uses **binary cross-entropy loss** and **RMSprop optimizer** to train the model.  
- **Performance Metrics**: Accuracy, confusion matrix, and **ROC-AUC** for model evaluation.  
- **Visualization**: Plots training/validation loss and accuracy to monitor performance.  

## **Model Architecture**  
| Layer | Output Shape | Parameters |  
|---|---|---|  
| Conv2D (32, 3x3) + ReLU | 148 x 148 x 32 | 896 |  
| MaxPooling2D (2x2) | 74 x 74 x 32 | 0 |  
| Conv2D (32, 3x3) + ReLU | 72 x 72 x 32 | 9248 |  
| MaxPooling2D (2x2) | 36 x 36 x 32 | 0 |  
| Conv2D (64, 3x3) + ReLU | 34 x 34 x 64 | 18496 |  
| MaxPooling2D (2x2) | 17 x 17 x 64 | 0 |  
| Flatten | 18496 | 0 |  
| Dense (64) + ReLU | 64 | 1183808 |  
| Dropout (0.5) | 64 | 0 |  
| Dense (1) + Sigmoid | 1 | 65 |  

## **Installation & Setup**  

### **1. Prerequisites**  
Ensure you have **Python** installed along with the required dependencies:  

```sh
pip install tensorflow keras numpy matplotlib pandas scikit-learn opencv-python
```

### **2. Dataset Preparation**  
- Store images in two directories:  
  - `Parasitized_1/` â Contains infected cell images.  
  - `Uninfected_1/` â Contains healthy cell images.  
- Resize all images to **150x150 pixels** before training.  

### **3. Run the Training Script**  
Execute the model training script:  

```sh
python train_model.py
```

### **4. Model Evaluation**  
To test the model on unseen data:  

```sh
python evaluate_model.py
```

## **Results**  
- **Training Accuracy**: ~90%  
- **Evaluation Metrics**:  
  - **Confusion Matrix** for error analysis.  
  - **ROC Curve** for optimal classification threshold.  
  - **AUC Score** for performance measurement.  

## **Future Enhancements**  
- **Hyperparameter Tuning** for better accuracy.  
- **Transfer Learning** using pre-trained models.  
- **Deployment** as a web-based malaria detection tool.  

## **Authors**  
- **Nandhini K (M230788EC)**   
National Institute of Technology Calicut, India  
