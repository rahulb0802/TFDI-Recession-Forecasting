# Extended Regularization Check - L1 Penalty
# Adapted from main_regularization_check_L1.py to use extended forecasting methodology
# Based on Rahul's Recursive_Forecasting.ipynb and main_extended_forecasting.py

# %% Importing libraries

import pandas as pd
import numpy as np
import os
import time
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %% Setting - path
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_PATH / 'results'
INTERMEDIATE_PATH = SCRIPT_PATH.parent / '03_intermediate_data'
OOS_PRED_PATH = RESULTS_PATH / 'oos_predictions'
SUB_INDICES_PATH = RESULTS_PATH / 'sub_indices_for_tuning'

# Create necessary directories
RESULTS_PATH.mkdir(exist_ok=True)
OOS_PRED_PATH.mkdir(exist_ok=True)
SUB_INDICES_PATH.mkdir(exist_ok=True)

# Machine Learning and Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

# Import forecasting tools
from fcst_tools import (
    generate_PCA_Factors,
    generate_TFDI_Sub_Indices,
    generate_Weakness_Indices,
    generate_Deter_Indices,
    add_lags,
    add_lags_wo_current,
    generate_qTrans_Sub_Indices,
    generate_TFDI_PCA_Factors
)

# Import additional forecasting tools
from fcst_tools_2 import (
    learn_in_sample_static_weights,
    select_top_variables_per_category,
    get_final_definitive_span,
    get_horizon_specific_optimal_span,
    generate_PCA_Factors_Binary,
    generate_TFDI_Sub_Indices_v2,
    generate_Weakness_Indices as generate_Weakness_Indices_v2,
    generate_Deter_Indices as generate_Deter_Indices_v2,
    generate_Deter_Avg_Indices,
    add_lags_wo_current
)

# %% Setting - config

warnings.filterwarnings("ignore")

SAVE_MODELS = True
OOS_MODELS_PATH = RESULTS_PATH / 'models'

# Out-of-sample (OOS) Loop Settings
OOS_START_DATE = '1990-01-01'
PREDICTION_HORIZONS = [6]
LAGS_TO_ADD = []

# Nonlinear grid for regularization parameters
ngrid = 30
# Create nonlinear grid from 0.001 to 1.0
C_values = np.logspace(-3, 1, ngrid)  # This creates 30 values from 0.001 to 1.0
print(f"Regularization grid (C values): {C_values}")

# Initialize variables for optimal C selection
optimal_C_values = {}  # Will store optimal C for each predictor set

# These are the models used to generate the ensemble forecasts with different regularization strengths
MODELS_TO_RUN = {}
for i, C in enumerate(C_values):
    model_name = f'Logit_L1_C{C:.3f}'
    MODELS_TO_RUN[model_name] = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=C,  # C is the inverse of regularization strength
        max_iter=1000, 
        random_state=42
    )

# Rerun Control
FORCE_RERUN_ALL_SETS = True
#FORCE_RERUN_SPECIFIC_SETS = ['Full', 'PCA_Factors_8']  # This is ignored if FORCE_RERUN_ALL_SETS is True
ALL_POSSIBLE_SETS = ['Deter_States_v2', 'Full', 'PCA_Factors_8']
#ALL_POSSIBLE_SETS = ['Deter_States_v2']
#ALL_POSSIBLE_SETS = ['Full', 'PCA_Factors_8', 'Deter_v2', 'Deter_States_v2', 'Deter_PCA_v2', 'Deter_Avg_v2']

# %% Load data
print("Loading analysis-ready datasets...")
y_target_full        = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'y_target.pkl'))
X_yield_full         = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_yield.pkl'))
X_transformed_full   = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_transformed_monthly.pkl'))
X_untransformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_untransformed_monthly.pkl'))
X_ads_full           = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_ads.pkl'))
tcodes               = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'tcodes.pkl'))
print("All data loaded successfully.")
print(f"Data shape: {X_transformed_full.shape}, Target shape: {y_target_full.shape}")

# %% Data filtering
VARS_TO_REMOVE = ['ACOGNO', 'TWEXAFEGSMTHx', 'UMCSENTx', 'OILPRICEx']

vars_that_exist_to_remove = [var for var in VARS_TO_REMOVE if var in X_transformed_full.columns]

X_transformed_full = X_transformed_full.drop(columns=vars_that_exist_to_remove)
X_untransformed_full = X_untransformed_full.drop(columns=vars_that_exist_to_remove)

print(f"--- Data Filtering Complete ---")
print(f"Removed {len(vars_that_exist_to_remove)} problematic variables from all master DataFrames.")
print(f"The list of removed variables is: {vars_that_exist_to_remove}")
print(f"New data shape: {X_transformed_full.shape}")

# %% Main Forecasting Loop
print("\n--- Starting regularization robustness check with extended methodology ---")

# Master Loop for All Horizons
for PREDICTION_HORIZON in PREDICTION_HORIZONS:
    print(f"\n{'='*25} Processing Horizon h={PREDICTION_HORIZON} {'='*25}")

    # Shift predictors to align with target
    X_transformed_shifted = X_transformed_full.shift(PREDICTION_HORIZON)
    X_untransformed_shifted = X_untransformed_full.shift(PREDICTION_HORIZON)
    X_yield_shifted = X_yield_full.shift(PREDICTION_HORIZON)
    X_ads_shifted = X_ads_full.shift(PREDICTION_HORIZON)

    file_path = OOS_PRED_PATH / f'regularization_results_h{PREDICTION_HORIZON}.pkl'

    horizon_sub_index_path = SUB_INDICES_PATH / f'h{PREDICTION_HORIZON}'
    horizon_sub_index_path.mkdir(exist_ok=True)

    if SAVE_MODELS:
        horizon_model_path = OOS_MODELS_PATH / f'h{PREDICTION_HORIZON}'
        horizon_model_path.mkdir(exist_ok=True)
        print(f"Models for h = {PREDICTION_HORIZON} will be saved to: {horizon_model_path}")

    # Initialize storage for results
    oos_probs, oos_errors, oos_actuals, oos_importances = {}, {}, None, {}
    model_coefficients = {}  # Store coefficients for each model
    model_intercepts = {}    # Store intercepts for each model

    # Determine which predictor sets need to be run
    sets_to_run = ALL_POSSIBLE_SETS
    print(f"The following {len(sets_to_run)} sets will be run: {sets_to_run}")

    # Initialize/clear storage for the sets that are being run
    for pred_set in sets_to_run:
        oos_probs[pred_set] = {m: [] for m in MODELS_TO_RUN}
        oos_errors[pred_set] = {m: [] for m in MODELS_TO_RUN}
        oos_importances[pred_set] = {m: [] for m in MODELS_TO_RUN}
        model_coefficients[pred_set] = {m: [] for m in MODELS_TO_RUN}
        model_intercepts[pred_set] = {m: [] for m in MODELS_TO_RUN}

    target_col_name = 'USRECM'

    # Main time-series loop
    all_dates = y_target_full.index
    forecast_dates = all_dates[all_dates >= pd.to_datetime(OOS_START_DATE)]

    if oos_actuals is None or (len(oos_actuals) != len(forecast_dates)):
        oos_actuals = y_target_full.loc[forecast_dates, target_col_name]
        oos_actuals.index.name = 'Date'

    start_time = time.time()

    for i, forecast_date in enumerate(forecast_dates):
        iter_start_time = time.time()
        train_end_date = forecast_date - pd.DateOffset(months=PREDICTION_HORIZON)
        y_train_full = y_target_full.loc[:train_end_date, target_col_name]
        y_actual = oos_actuals.loc[forecast_date]

        print(f"Iter {i+1}/{len(forecast_dates)}: h={PREDICTION_HORIZON}, Date={forecast_date.date()}... ", end="")

        # --- Centralized Data Preparation ---
        X_train_transformed_slice = X_transformed_shifted.loc[:forecast_date].copy()
        X_untransformed_slice = X_untransformed_shifted.loc[:forecast_date].copy()
        X_train_valid_cols = X_train_transformed_slice.drop(columns=X_train_transformed_slice.columns[X_train_transformed_slice.isna().all()]).copy()
        imputer_base = KNNImputer(n_neighbors=5)
        X_train_imputed = pd.DataFrame(imputer_base.fit_transform(X_train_valid_cols), index=X_train_valid_cols.index, columns=X_train_valid_cols.columns)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), index=X_train_imputed.index, columns=X_train_imputed.columns)

        # Generate Predictor Sets
        predictor_data_iter = {}

        if 'ADS' in sets_to_run:
            predictor_data_iter['ADS'] = X_ads_shifted.loc[:forecast_date]
        if 'Yield' in sets_to_run:
            predictor_data_iter['Yield'] = X_yield_shifted.loc[:forecast_date]
        if 'Full' in sets_to_run: 
            predictor_data_iter['Full'] = X_train_scaled
        if 'PCA_Factors_8' in sets_to_run:
            # The function does all preprocessing internally
            all_8_factors = generate_PCA_Factors(X_train_transformed_slice, n_factors=8)
            predictor_data_iter['PCA_Factors_8'] = all_8_factors
        if 'Deter_v2' in sets_to_run:
            predictor_data_iter['Deter_v2'], avg_weights, _ = generate_Deter_Indices_v2(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
        if 'Deter_States_v2' in sets_to_run:
            _, _, predictor_data_iter['Deter_States_v2'] = generate_Deter_Indices_v2(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
        if 'Deter_PCA_v2' in sets_to_run:
            _, _, matrix = generate_Deter_Indices_v2(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
            predictor_data_iter['Deter_PCA_v2'] = generate_PCA_Factors_Binary(matrix, n_factors=8)
        if 'Deter_Avg_v2' in sets_to_run:
            predictor_data_iter['Deter_Avg_v2'], _ = generate_Deter_Avg_Indices(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)

        # Loop over the INTENDED sets
        for pred_set_name in sets_to_run:
            X_train_raw = predictor_data_iter.get(pred_set_name)

            if X_train_raw is None or X_train_raw.empty:
                for model_name in MODELS_TO_RUN:
                    oos_probs[pred_set_name][model_name].append(np.nan)
                    oos_errors[pred_set_name][model_name].append(np.nan)
                    model_coefficients[pred_set_name][model_name].append(np.nan)
                    model_intercepts[pred_set_name][model_name].append(np.nan)
                continue
            else:
                X_train_lagged = add_lags(X_train_raw, LAGS_TO_ADD, prefix=f'{pred_set_name}_')

            X_train_lagged_final = X_train_lagged.dropna()
            X_train_final = X_train_lagged_final.loc[:train_end_date]
            X_predict_point = X_train_lagged_final.loc[[forecast_date]]

            for model_name, model_template in MODELS_TO_RUN.items():
                prob, error = np.nan, np.nan
                coefficients, intercept = np.nan, np.nan
                importances = pd.Series()

                try:
                    common_index = y_train_full.index.intersection(X_train_final.index)
                    y_train_aligned = y_train_full.loc[common_index]
                    X_train_aligned = X_train_final.loc[common_index]

                    if len(X_train_aligned) > max(LAGS_TO_ADD, default=0) + 20:
                        model_instance = clone(model_template)

                        # Set class weight to balanced
                        model_instance = model_template.set_params(class_weight='balanced')
                        #model_instance = model_template.set_params(class_weight=None)

                        X_predict_imputed = X_predict_point.reindex(columns=X_train_aligned.columns).ffill().bfill()

                        if not X_predict_imputed.isna().any().any():
                            model_instance.fit(X_train_aligned, y_train_aligned)

                            # Store model parameters
                            if hasattr(model_instance, 'coef_'):
                                coefficients = model_instance.coef_[0]
                                importances = pd.Series(np.abs(model_instance.coef_[0]), index=X_train_aligned.columns)
                            if hasattr(model_instance, 'intercept_'):
                                intercept = model_instance.intercept_[0]

                            if SAVE_MODELS and forecast_date == forecast_dates[-1]:
                                model_filename = f'{pred_set_name}_{model_name}.pkl'
                                model_path = horizon_model_path / model_filename
                                joblib.dump(model_instance, model_path)

                            prob = model_instance.predict_proba(X_predict_imputed)[:, 1][0]
                            error = (y_actual - prob)**2
                except Exception as e:
                    print(f"Error in model {model_name} for set {pred_set_name}: {e}")
                    pass

                oos_probs[pred_set_name][model_name].append(prob)
                oos_errors[pred_set_name][model_name].append(error)
                oos_importances[pred_set_name][model_name].append(importances)
                model_coefficients[pred_set_name][model_name].append(coefficients)
                model_intercepts[pred_set_name][model_name].append(intercept)

        iter_end_time = time.time()
        print(f" ({(iter_end_time - iter_start_time):.2f}s)")

    # Save results after each horizon's loop is complete
    print(f"\n--- Loop for h={PREDICTION_HORIZON} Finished ---")
    results_to_save = {
        'probabilities': oos_probs, 
        'squared_errors': oos_errors, 
        'actuals': oos_actuals, 
        'importances': oos_importances,
        'model_coefficients': model_coefficients,
        'model_intercepts': model_intercepts,
        'C_values': C_values,
        'prediction_horizon': PREDICTION_HORIZON
    }
    joblib.dump(results_to_save, file_path)
    print(f"Updated results saved to: {file_path}")

print("\n--- All Horizons Complete ---")

# %% Report results
end_time = time.time()
print(f"Total computation time: {end_time - start_time:.2f} seconds")
print("")

# Calculate and report average Brier scores (OOS errors)
print("=" * 80)
print("AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)")
print("=" * 80)

# Create a summary DataFrame for better visualization
results_summary = []

for pred_set_name in sets_to_run:
    print(f"\nPredictor Set: {pred_set_name}")
    print("-" * 50)
    
    for model_name in MODELS_TO_RUN.keys():
        errors = oos_errors[pred_set_name][model_name]
        # Filter out NaN values for calculation
        valid_errors = [e for e in errors if not np.isnan(e)]
        
        if valid_errors:
            avg_brier = np.mean(valid_errors)
            n_valid = len(valid_errors)
            n_total = len(errors)
            print(f"  {model_name:20s}: {avg_brier:.6f} (n={n_valid}/{n_total})")
            
            results_summary.append({
                'Predictor_Set': pred_set_name,
                'Model': model_name,
                'C_Value': float(model_name.split('_C')[1]),
                'Avg_Brier_Score': avg_brier,
                'Valid_Predictions': n_valid,
                'Total_Predictions': n_total
            })
        else:
            print(f"  {model_name:20s}: No valid predictions")
            
            results_summary.append({
                'Predictor_Set': pred_set_name,
                'Model': model_name,
                'C_Value': float(model_name.split('_C')[1]),
                'Avg_Brier_Score': np.nan,
                'Valid_Predictions': 0,
                'Total_Predictions': len(errors)
            })

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results_summary)
print(f"\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(results_df.to_string(index=False, float_format='%.6f'))

# Find best performing model for each predictor set
print(f"\n" + "=" * 80)
print("BEST PERFORMING MODELS BY PREDICTOR SET")
print("=" * 80)

for pred_set in sets_to_run:
    subset = results_df[results_df['Predictor_Set'] == pred_set]
    valid_subset = subset.dropna(subset=['Avg_Brier_Score'])
    
    if not valid_subset.empty:
        best_model = valid_subset.loc[valid_subset['Avg_Brier_Score'].idxmin()]
        print(f"{pred_set:15s}: {best_model['Model']:20s} (C={best_model['C_Value']:.3f}, Brier: {best_model['Avg_Brier_Score']:.6f})")
    else:
        print(f"{pred_set:15s}: No valid predictions")

print("")

# %% Save results
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save summary table as CSV
summary_filename = f'regularization_summary_extended_h{PREDICTION_HORIZONS[0]}.csv'
results_df.to_csv(RESULTS_PATH / summary_filename, index=False)
print(f"Saved summary table to: {RESULTS_PATH / summary_filename}")

print("Results saved successfully!")

# %% Plot regularization parameter vs average Brier score
fig, ax = plt.subplots(figsize=(12, 8))

for pred_set in sets_to_run:
    subset = results_df[(results_df['Predictor_Set'] == pred_set) &
                        (~results_df['Avg_Brier_Score'].isna())]
    if not subset.empty:
        ax.plot(subset['C_Value'],
                subset['Avg_Brier_Score'],
                marker='o',
                label=pred_set,
                linewidth=2,
                markersize=6)

ax.set_xscale('log')
ax.set_xlabel('C value (inverse regularization strength)', fontsize=12)
ax.set_ylabel('Average Brier score', fontsize=12)
ax.set_title('Regularization Grid vs Brier Score - Extended Methodology', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

if len(sets_to_run) > 1:
    ax.legend(fontsize=10)

plt.tight_layout()
fig_filename = RESULTS_PATH / 'fig_reg_extended.png'
fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
print(f"Saved regularization plot to: {fig_filename}")
plt.show()

# %% Additional Analysis: Coefficient Sparsity Analysis
print("\n" + "=" * 80)
print("COEFFICIENT SPARSITY ANALYSIS")
print("=" * 80)

# Analyze coefficient sparsity for different regularization strengths
sparsity_summary = []

for pred_set_name in sets_to_run:
    print(f"\nPredictor Set: {pred_set_name}")
    print("-" * 50)
    
    for model_name in MODELS_TO_RUN.keys():
        coefficients_list = model_coefficients[pred_set_name][model_name]
        # Filter out NaN coefficients
        valid_coefficients = [coef for coef in coefficients_list if isinstance(coef, np.ndarray) and not np.isnan(coef).any()]
        
        if valid_coefficients and len(valid_coefficients) > 0:
            # Calculate average sparsity (proportion of zero coefficients)
            sparsity_per_iter = []
            for coef_array in valid_coefficients:
                if isinstance(coef_array, np.ndarray):
                    zero_coefs = np.sum(np.abs(coef_array) < 1e-10)  # Near-zero threshold
                    total_coefs = len(coef_array)
                    sparsity = zero_coefs / total_coefs
                    sparsity_per_iter.append(sparsity)
            
            if sparsity_per_iter:
                avg_sparsity = np.mean(sparsity_per_iter)
                print(f"  {model_name:20s}: {avg_sparsity:.3f} avg sparsity")
                
                sparsity_summary.append({
                    'Predictor_Set': pred_set_name,
                    'Model': model_name,
                    'C_Value': float(model_name.split('_C')[1]),
                    'Avg_Sparsity': avg_sparsity
                })

# Plot sparsity vs C values
if sparsity_summary:
    sparsity_df = pd.DataFrame(sparsity_summary)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for pred_set in sets_to_run:
        subset = sparsity_df[sparsity_df['Predictor_Set'] == pred_set]
        if not subset.empty:
            ax.plot(subset['C_Value'],
                    subset['Avg_Sparsity'],
                    marker='s',
                    label=pred_set,
                    linewidth=2,
                    markersize=6)
    
    ax.set_xscale('log')
    ax.set_xlabel('C value (inverse regularization strength)', fontsize=12)
    ax.set_ylabel('Average Coefficient Sparsity', fontsize=12)
    ax.set_title('Regularization vs Coefficient Sparsity - Extended Methodology', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if len(sets_to_run) > 1:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    sparsity_fig_filename = RESULTS_PATH / 'fig_sparsity_extended.png'
    fig.savefig(sparsity_fig_filename, dpi=300, bbox_inches='tight')
    print(f"Saved sparsity plot to: {sparsity_fig_filename}")
    plt.show()

print("\n--- Extended Regularization Analysis Complete ---")
