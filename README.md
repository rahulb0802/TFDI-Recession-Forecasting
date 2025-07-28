# A Deterioration Diffusion Index for U.S. Recession Forecasting
**Author:** Rahul Billakanti

## Project Overview
This repository contains the code and data to replicate the results in the paper "A Deterioration Diffusion Index for U.S. Recession Forecasting". The project develops and tests a novel diffusion index for forecasting U.S. recessions.

## Required Libraries
All necessary Python libraries are listed in `requirements.txt`. To create the correct environment, run:
`pip install -r requirements.txt`

## Instructions for Reproducing Results

To generate all results from scratch, please run the Jupyter/Colab notebooks in the following numerical order. It is recommended to use "Runtime -> Restart and run all" for each notebook to ensure a clean execution.

**Step 1: Data Preparation**
- **File:** `1_Data_Preparation.ipynb`
- **Purpose:** Downloads the latest FRED-MD data, performs initial cleaning, and saves the base data files to the `03_intermediate_data` folder.
- **Expected Runtime:** ~1-2 minutes.

**Step 2: Recursive Forecasting**
- **File:** `2_Recursive_Forecasting.ipynb`
- **Purpose:** Runs the main out-of-sample forecasting loop for all models (TFDI, PCA, ADS, etc.) for both the h=1 and h=3 horizons. This is a computationally intensive script.
- **Configuration:** Before running, ensure the `FORCE_RERUN_ALL` variable in the configuration cell is set to `True` to generate results from scratch.
- **Expected Runtime:** ~4-6 hours.

**Step 3: Evaluation and Analysis**
- **File:** `3_Evaluation_and_Analysis.ipynb`
- **Purpose:** Loads the saved forecast results (`.pkl` files) and generates all tables and figures used in the final paper.
- **Configuration:** The `PREDICTION_HORIZON` variable in the configuration cell can be set to `1` or `3` to generate the results for the nowcast or forecast, respectively. The notebook must be run once for each horizon.
- **Expected Runtime:** ~2-4 minutes per horizon.
