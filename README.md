# HyMLRaman
Hybrid Machine Learning-Based Raman Spectroscopy with Generative Augmentation for Low-Data Pharmaceutical Identification

<img width="1661" height="997" alt="image" src="https://github.com/user-attachments/assets/346b3bc7-7e9b-4c81-bf6a-5f5e672c67a4" />


# Requirement
HyMLRaman requires the packages as follows: 
- `torch`: an open-source machine learning library with strong GPU acceleration.
- `torchvision`: a popular PyTorch package that provides computer vision utilities, popular datasets, model architectures, and image transformations.
- `jupyterlab`: a web-based interactive development environment for notebooks, code, and data.
- `matplotlib`: a comprehensive library for creating static, animated, and interactive visualizations in Python.
- `scikit-learn`: a set of Python modules for machine learning and data mining
- `python-docx`: a Python library for creating and updating Microsoft Word (.docx) files
- `xgboost`: a machine learning library based on the gradient-boosted decision trees algorithm.
- `openTSNE`: a modular Python implementation of t-Distributed Stochastic Neighbor Embedding (t-SNE).
- `umap-learn`: a dimension reduction technique that can be used for the UMAP method.

Example to install requirements with conda for CPU & GPU:
```md
$ conda create -n torch python=3.9
$ conda activate torch
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$ pip3 install jupyterlab matplotlib seaborn scikit-learn python-docx imagehash opencv-python xgboost openTSNE umap-learn streamlit
```

# Directory Description

```md
HyMLRaman
├── data
│   └── docx: a dataset of Raman spectra of 6 pharmaceuticals (total 1003 samples), which humans collected from the literature.
│   └── test: the unseen experimental Raman spectra of 6 pharmaceuticals (in-house Raman measurement) for testing the app.
├── utils
│   ├── utils_data.py: define the functions related to the dataset.
│   ├── utils_model.py: define the function related to the model.
│   └── utils_plot.py: defines the function related to plotting results.
│   └── utils_train.py: defines the function related to training.
│   └── utils_feature.py: defines the function related to feature extraction.
│   └── utils_ddpm.py: defines the function related to DDPM.
├── model
│   ├── resnet18_10cls_20251115.pth: a trained GNN model.
├── output
│   ├── xgb_on_cnn.pkl: a trained XGBoost model with GNN features and PCA.
├── HyMLRaman.ipynb: Main MLRaman code
└── app.py: user-friendly application for real-time prediction based on PyQt6.
```
# How to run
Step 1: Download the MLRaman package:

    git clone https://github.com/nguyen-group/MLRaman.git

Step 2: Go to the source code in the Raman directory to run the program:

    cd MLRaman
    jupyter-lab MLRaman.ipynb

Step 3: For Streamlit application:

    streamlit run app_xgb_streamlit.py

Note: `app_xgb_streamlit.py` will load `resnet18_10cls_20251115.pth` and `xgb_on_cnn.pkl`, which are stored in the model and output directories, respectively.

# References and citing
The detailed MLRaman is described in our paper:
> Q. T. T. Binh, L. T. Phuoc, P. X. Hai, T. B. Phan, V. T. H. Thu* and N. T. Hung*, Rapid machine learning-driven detection of pesticides and dyes using Raman spectroscopy, J. Chem. Inf. Model. 66, 3803-3813 (2026).
> 
> [https://doi.org/10.1021/acs.jcim.6c00396](https://doi.org/10.1021/acs.jcim.6c00396)
