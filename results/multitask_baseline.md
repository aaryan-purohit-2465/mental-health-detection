# Multitask Baseline Results

**Date:** 22 Nov 2025  
**Model:** DistilBERT (multitask heads: MH + Emotion)  
**Dataset:** synthetic 20 samples  
**Config:** epochs=2, batch_size=8, lr=2e-5, multitask=true, emo_weight=0.4

### Loss per epoch
- Epoch 1: 4.0292
- Epoch 2: 3.1893

### Notes
- Tiny synthetic set — numbers are only to confirm pipeline works.
- Checkpoint saved locally at `checkpoints/demo_model/`.
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("Classification Report:\n")

# Convert predictions + labels to numpy arrays
y_true = np.array(labels)
y_pred = np.array(preds)

# Print detailed classification metrics
print(classification_report(y_true, y_pred, digits=2))

# Confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
