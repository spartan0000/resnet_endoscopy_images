# ResNet-50 Cecum Detection in Endoscopy Images

Fine-tunes an ImageNet-pretrained ResNet-50 to perform **binary classification** on lower-GI-tract endoscopy images: *cecum* vs. *not cecum*. Training is done in [training.py](training.py).

## Data

- Dataset: the lower-GI-tract labeled images from [HyperKvasir](https://datasets.simula.no/hyper-kvasir/), loaded with `torchvision.datasets.ImageFolder` (one sub-folder per class).
- The path is hard-coded in `training.py` (`root=...`); edit it to point to your local copy.
- Multi-class folder labels are collapsed into a binary target (`1` = cecum, `0` = everything else).

### Split

- 80% train / 20% validation, using scikit-learn's `train_test_split` **stratified on the binary cecum label** with `random_state=42`. Both sets therefore contain cecum images in the same proportion, and the split is reproducible between runs.
- Two `ImageFolder` instances (one per transform) are wrapped in `Subset`s using the same indices, so training and validation get different transforms without overlapping.

### Transforms

| | Training | Validation |
| --- | --- | --- |
| Resize | 224×224 | 224×224 |
| Augmentation | random horizontal flip, random vertical flip, random rotation (±20°) | none |
| Normalization | ImageNet mean/std | ImageNet mean/std |

## Handling class imbalance

Cecum images are a minority. The training loader uses a `WeightedRandomSampler` (with replacement, one epoch = number of training samples) with inverse-class-frequency weights, so each batch is drawn roughly evenly from both classes. The loss itself is unweighted (`pos_weight` is computed in the script but not currently passed to the loss).

## Model

- `torchvision` ResNet-50 with `IMAGENET1K_V2` weights.
- All parameters are frozen, then **`layer4`** (the last residual block) is unfrozen for fine-tuning.
- The original 1000-class `fc` layer is replaced with: `Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→1)`. The head outputs a raw logit; the sigmoid is applied inside the loss (training) and explicitly at evaluation time.

## Training setup

| Setting | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 32 (train and validation) |
| Optimizer | Adam, lr = 1e-5 (trainable params only) |
| Loss | `BCEWithLogitsLoss` (no `pos_weight`) |
| Label smoothing | Manual, α = 0.1: training targets become `y·0.9 + 0.05` (1 → 0.95, 0 → 0.05). Validation uses hard labels |
| Scheduler | `ReduceLROnPlateau` on validation loss (factor 0.5, patience 3) |
| Precision | Mixed precision (`autocast` + `GradScaler`) |
| Device | CUDA if available, otherwise CPU |
| DataLoader workers | 4, `pin_memory=True` |

## Per-epoch output

Each epoch runs a training pass, then a validation pass (no gradients), and prints:

- training loss and validation loss (averaged over batches; note the training loss uses smoothed labels, so the two are not directly comparable)
- validation **accuracy** (threshold: sigmoid output > 0.5)
- validation **sensitivity** (recall for cecum: TP / (TP + FN))
- current learning rate

Example line:

```
Epoch 1/20 | Train loss: ... | Val loss: ... | Accuracy: ... | Sensitivity: ... | LR: 1.00e-05
```

At startup nothing is printed beyond this; each time the validation loss improves, `Best model saved at epoch N` is printed and the weights are written to `best_model.pth`.

## Running

```
python training.py
```

Requires `torch`, `torchvision` and `scikit-learn`. The training loop is wrapped in `training_loop()` and runs under `if __name__ == "__main__":`, which is needed for the multi-worker DataLoaders on Windows.
