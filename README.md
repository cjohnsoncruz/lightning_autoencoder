# Neural Activity Autoencoder

A PyTorch Lightning autoencoder for analyzing latent structure in calcium imaging data from mouse prefrontal cortex during behavioral tasks.

## Overview

This project implements an autoencoder architecture using PyTorch Lightning for dimensionality reduction and analysis of neural population activity. The framework supports processing, modeling, and visualization of calcium imaging timeseries data across different task phases and experimental conditions.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/neural-activity-autoencoder.git
cd neural-activity-autoencoder

# Install dependencies
pip install -r autoencoder_cloud_requirements.txt
```

## Project Structure

```text
neural-activity-autoencoder/
├── autoencoder_training.ipynb       # Main notebook for training and analysis
├── external_functions.py            # Utility functions for data processing
├── ax_modifier_functions_cloud.py   # Matplotlib axis customization
├── sns_plotting_config.py           # Seaborn plotting configuration
├── paper_plot.mplstyle              # Matplotlib style for publication figures
├── autoencoder_cloud_requirements.txt
├── data/                            # Input data (calcium timeseries, ensembles)
├── models/                          # Trained model checkpoints
└── output/                          # Generated figures and analysis results
```

## Usage

The main analysis notebook `autoencoder_training.ipynb` contains:

- Loading and preprocessing calcium timeseries data
- Training autoencoder models with PyTorch Lightning
- Visualizing latent representations
- Analyzing neural population activity across task phases

## Visualization

The project uses matplotlib and seaborn for generating publication-quality figures:

- `ax_modifier_functions_cloud.py`: Matplotlib axis formatting utilities
- `sns_plotting_config.py`: Consistent seaborn styling
- `paper_plot.mplstyle`: Publication-ready figure formatting
