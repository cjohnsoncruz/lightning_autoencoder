# Neural Activity Autoencoder

A PyTorch Lightning-based autoencoder framework for analyzing neural activity patterns during task phases, with built-in visualization tools and cloud deployment support.

## 📋 Overview

This project implements an autoencoder architecture using PyTorch Lightning for dimensionality reduction and analysis of neural activity data. The framework supports processing, modeling, and visualization of calcium imaging timeseries data across different task phases and conditions.

## 🔧 Installation

```bash
# Clone the repository
git clone https://your-repository-url/lightning_studio_autoencoder.git
cd lightning_studio_autoencoder

# Install dependencies
pip install -r autoencoder_cloud_requirements.txt
```
## 📦 Current Project Structure
```
lightning_studio_autoencoder/
│
├── .ipynb_checkpoints/          # Jupyter notebook checkpoints
├── __pycache__/                 # Python cache files
├── autoencoder_cloud_requirements.txt  # Project dependencies
├── ax_modifier_functions_cloud.py     # Matplotlib axis customization functions
├── cloudrun_autoenc_lightning_v4.ipynb # Main notebook for autoencoder analysis
├── data/                        # Data directory
│   ├── Dlx56_Normalized Trial Calcium Timeseries_20_Jun_2025.parquet
│   ├── ensemble_resample_dict_500samples_20_Jun_2025.pickle
│   └── pseudopopulation_ensemble_activity_store_20_Jun_2025.pickle
├── external_functions.py        # Utility functions for data processing and analysis
├── figure_output/               # Directory for generated figures
├── lightning_logs/             # PyTorch Lightning log directory
├── paper_plot.mplstyle         # Matplotlib style for publication-quality figures
├── sns_plotting_config.py       # Seaborn plotting configuration
└── README.md                    # This file
```
## 🚀 Usage
### Running Analysis Notebook

The main analysis notebook `cloudrun_autoenc_lightning_v4.ipynb` contains comprehensive examples for:
- Loading and preprocessing calcium timeseries data
- Training autoencoder models
- Visualizing latent representations
- Analyzing neural population activity across task phases

## 📊 Visualization

The project uses matplotlib and seaborn for visualization.
- Matplotlib-based plotting utilities (`ax_modifier_functions_cloud.py`)
- Seaborn configuration for consistent styling (`sns_plotting_config.py`)
- Functions for generating publication-quality figures

Data samples are stored in the `data/` directory.
