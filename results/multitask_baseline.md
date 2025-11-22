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
