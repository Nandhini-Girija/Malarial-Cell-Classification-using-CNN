from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
X_test = np.load('X_test.npy')  # Assuming pre-saved test data
y_test = np.load('y_test.npy')

# Load trained model
model = load_model('malaria_model.h5')

# Evaluate accuracy
_, acc = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Generate predictions
y_preds = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_preds)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Confusion matrix
threshold = 0.5
y_pred_labels = (y_preds >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:
", cm)

# Compute AUC
auc_score = auc(fpr, tpr)
print(f"Area Under Curve (AUC): {auc_score:.3f}")
