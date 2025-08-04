# Rahul' Recursive_Forecasting.ipynb 
# Minchul's replication


# %% Importing libraries
import os
import pandas as pd
import numpy as np
import time

# %% Setting - path
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_PATH / 'results'
INTERMEDIATE_PATH = SCRIPT_PATH.parent / '03_intermediate_data'

# sklearn models and utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.base import clone


# Import forecasting tools
from fcst_tools import (
    generate_PCA_Factors,
    generate_TFDI_Sub_Indices,
    generate_Weakness_Indices,
    generate_Deter_Indices,
    add_lags,
    add_lags_wo_current,
)

# %% Setting - config

SAVE_MODELS = True
OOS_MODELS_PATH = RESULTS_PATH  # dump everything to this folder

# Out-of-sample (OOS) Loop Settings
OOS_START_DATE = '1990-01-01'
PREDICTION_HORIZONS = [3]
LAGS_TO_ADD = []

# These are the five models used to generate the ensemble forecasts
MODELS_TO_RUN = {
    'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
    'Logit_L1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),

    # original
    #'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
    #'Logit_L1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
    #'HGBoost': HGBoost(random_state=42),
    #'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    #'RandomForest': RandomForestClassifier(random_state=42),
}

# Predictors
#ALL_POSSIBLE_SETS = ['TFDI', 'PCA_Factors_8', 'Full']
ALL_POSSIBLE_SETS = ['PCA_Factors_8', 'Full', 'Yield', 'ADS']

# %% Load data
y_target_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'y_target.pkl'))
X_yield_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_yield.pkl'))
X_transformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_transformed_monthly.pkl'))
X_untransformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_untransformed_monthly.pkl'))
X_ads_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_ads.pkl'))
tcodes = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'tcodes.pkl'))

# %% Actual computation
# Master Loop for all horizons
oos_probs = {}
oos_errors = {}
for PREDICTION_HORIZON in PREDICTION_HORIZONS:
    
    # Shift predictor to aligned with the target (y__{t+h} <- x_{t})
    X_transformed_full = X_transformed_full.shift(PREDICTION_HORIZON)
    X_untransformed_full = X_untransformed_full.shift(PREDICTION_HORIZON)
    X_yield_full = X_yield_full.shift(PREDICTION_HORIZON)
    X_ads_full = X_ads_full.shift(PREDICTION_HORIZON)

    # Determine which predictor sets need to be run
    sets_to_run = []
    sets_to_run = ALL_POSSIBLE_SETS

    # Initialize storage for results
    for pred_set in sets_to_run:
        oos_probs[pred_set] = {m: [] for m in MODELS_TO_RUN}
        oos_errors[pred_set] = {m: [] for m in MODELS_TO_RUN}

    # Main time-series loop
    all_dates = y_target_full.index
    forecast_dates = all_dates[all_dates >= pd.to_datetime(OOS_START_DATE)]

    oos_actuals = y_target_full.loc[forecast_dates, 'USRECM']
    oos_actuals.index.name = 'Date'

    start_time = time.time()

    for i, forecast_date in enumerate(forecast_dates):
        iter_start_time = time.time()

        # --- Prepare training data ---
        #train_end_date = forecast_date - pd.DateOffset(months=PREDICTION_HORIZON)
        train_end_date = forecast_date - pd.DateOffset(months=1) #note that we've already shifted X so our training is all but the last row
        y_train_full = y_target_full.loc[:train_end_date, 'USRECM']
        y_actual = oos_actuals.loc[forecast_date]
    
        print(f"Iter {i+1}/{len(forecast_dates)}: h={PREDICTION_HORIZON}, Date={forecast_date.date()}... ", end="")
        print("")

        # --- Centralized Data Preparation ---
        X_train_transformed_slice = X_transformed_full.loc[:forecast_date].copy()
        X_untransformed_slice = X_untransformed_full.loc[:forecast_date].copy()
        X_train_valid_cols = X_train_transformed_slice.drop(columns=X_train_transformed_slice.columns[X_train_transformed_slice.isna().all()]).copy()
        imputer_base = KNNImputer(n_neighbors=5)
        X_train_imputed = pd.DataFrame(imputer_base.fit_transform(X_train_valid_cols), index=X_train_valid_cols.index, columns=X_train_valid_cols.columns)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), index=X_train_imputed.index, columns=X_train_imputed.columns)
    
        # Generate Predictor Sets
        predictor_data_iter = {}

        if 'ADS' in sets_to_run:
            predictor_data_iter['ADS'] = X_ads_full.loc[:forecast_date]
        if 'Yield' in sets_to_run:
            predictor_data_iter['Yield'] = X_yield_full.loc[:forecast_date]
        if 'Full' in sets_to_run: predictor_data_iter['Full'] = X_train_scaled
        if 'PCA_Factors_8' in sets_to_run:
            # The function does all preprocessing internally
            all_8_factors = generate_PCA_Factors(X_train_transformed_slice, n_factors=8)
            predictor_data_iter['PCA_Factors_8'] = all_8_factors
        if 'TFDI' in sets_to_run:
            predictor_data_iter['TFDI'], all_sub_indices = generate_TFDI_Sub_Indices(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
            sub_index_filename = f'sub_indices_TFDI_{forecast_date.strftime("%Y-%m-%d")}.pkl'
        if 'Weakness' in sets_to_run:
            predictor_data_iter['Weakness'], all_sub_indices = generate_Weakness_Indices(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
            sub_index_filename = f'sub_indices_Weakness_{forecast_date.strftime("%Y-%m-%d")}.pkl'
        if 'Deter' in sets_to_run:
            predictor_data_iter['Deter'], all_sub_indices = generate_Deter_Indices(X_train_imputed, y_train_full, horizon=PREDICTION_HORIZON)
            sub_index_filename = f'sub_indices_Deter_{forecast_date.strftime("%Y-%m-%d")}.pkl'

        # Loop over the Intended sets
        for pred_set_name in sets_to_run:
            X_train_raw = predictor_data_iter.get(pred_set_name)
            if X_train_raw is None or X_train_raw.empty:
                for model_name in MODELS_TO_RUN:
                    oos_probs[pred_set_name][model_name].append(np.nan)
                    oos_errors[pred_set_name][model_name].append(np.nan)
                
                continue
            else:
                X_train_lagged = add_lags(X_train_raw, LAGS_TO_ADD, prefix=f'{pred_set_name}_')

            X_train_lagged_final = X_train_lagged.dropna()
            X_train_final = X_train_lagged_final.loc[:train_end_date]
            X_predict_point = X_train_lagged_final.loc[[forecast_date]]
            
            for model_name, model_template in MODELS_TO_RUN.items():
                prob, error = np.nan, np.nan
                try:
                    common_index = y_train_full.index.intersection(X_train_final.index)
                    y_train_aligned = y_train_full.loc[common_index]
                    X_train_aligned = X_train_final.loc[common_index]

                    # Filling missing values (if any, Minchul doubts) -> we shouldn't do it for now
                    # X_predict_reindexed = X_predict_point.reindex(columns=X_train_aligned.columns)
                    # missing_count = X_predict_reindexed.isna().sum().sum()
                    # X_predict_imputed = X_predict_reindexed.fillna(X_train_aligned.mean())
                    
                    # # not sure if we need this imputation; checking if there is a case that we are filling NAs
                    # if missing_count > 0:
                    #     print(f"  Imputed {missing_count} missing values for {model_name}/{pred_set_name}")
                    # else:
                    #     print(f"  No imputation needed for {model_name}/{pred_set_name}")
                    X_predict_imputed = X_predict_point


                    # Estimation
                    model_instance = clone(model_template)
                    fitted = model_instance.fit(X_train_aligned, y_train_aligned)
                    prob = model_instance.predict_proba(X_predict_imputed)[:, 1][0]
                    error = (y_actual - prob)**2 #Brier score

                except Exception as e:
                    print(f"Error in model {model_name} for set {pred_set_name} at date {forecast_date}: {e}")
            
                oos_probs[pred_set_name][model_name].append(prob)
                oos_errors[pred_set_name][model_name].append(error)

        iter_end_time = time.time()
        print(f"  Iteration time: {iter_end_time - iter_start_time:.2f} seconds")
        print("")


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
            print(f"  {model_name:15s}: {avg_brier:.6f} (n={n_valid}/{n_total})")
            
            results_summary.append({
                'Predictor_Set': pred_set_name,
                'Model': model_name,
                'Avg_Brier_Score': avg_brier,
                'Valid_Predictions': n_valid,
                'Total_Predictions': n_total
            })
        else:
            print(f"  {model_name:15s}: No valid predictions")
            
            results_summary.append({
                'Predictor_Set': pred_set_name,
                'Model': model_name,
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
        print(f"{pred_set:15s}: {best_model['Model']:15s} (Brier: {best_model['Avg_Brier_Score']:.6f})")
    else:
        print(f"{pred_set:15s}: No valid predictions")

print("")


# %%
