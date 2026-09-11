"""
Utility package for Raman image classification / deep-feature ML pipeline.

Modules:
- utils_env: environment, device, seed helpers
- utils_data: dataset construction, transforms, dataloaders
- utils_model: CNN/Vit models, losses, SAM, SpectraCNN wrapper
- utils_train: training loops, mixup, scheduler, TTA, K-fold
- utils_feature: CNN feature extraction and embedding/file-index helpers
- utils_ml: classical ML CV, ROC/CM utilities
- utils_ddpm: DDPM feature-level augmentation
- utils_check: data leakage checks
- utils_plot: paper-style figures
"""

__version__ = "0.1.0"
