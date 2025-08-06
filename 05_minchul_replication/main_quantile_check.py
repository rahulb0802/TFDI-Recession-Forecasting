# Robustness exercise for different h_qt and q_qt values

# %% Importing libraries
import os
import pandas as pd
import numpy as np
import time
import itertools
from pathlib import Path

# %% Setting - path
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
    generate_qTrans_Sub_Indices,
    generate_TFDI_PCA_Factors
)

# %% Setting - config
SAVE_MODELS = True
OOS_MODELS_PATH = RESULTS_PATH

# Out-of-sample (OOS) Loop Settings
OOS_START_DATE = '1990-01-01'
PREDICTION_HORIZONS = [1]
LAGS_TO_ADD = []

# Robustness parameters
H_QT_VALUES = [3]
Q_QT_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

# These are the models used to generate the ensemble forecasts
MODELS_TO_RUN = {
    'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
}

# Predictor sets to test
ALL_POSSIBLE_SETS = ['TFDI_Full_pca', 'TFDI_pca', 'TFDI_avg']

# %% Load data
y_target_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'y_target.pkl'))
X_yield_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_yield.pkl'))
X_transformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_transformed_monthly.pkl'))
X_untransformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_untransformed_monthly.pkl'))
X_ads_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_ads.pkl'))
tcodes = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'tcodes.pkl'))

# %% Function to run single robustness test
def run_robustness_test(h_qt, q_qt, pred_set_name):
    """
    Run a single robustness test with specified h_qt and q_qt values
    """
    print(f"\n{'='*60}")
    print(f"Running robustness test: h_qt={h_qt}, q_qt={q_qt}, pred_set={pred_set_name}")
    print(f"{'='*60}")
    
    # Initialize storage for results
    oos_probs = {pred_set_name: {m: [] for m in MODELS_TO_RUN}}
    oos_errors = {pred_set_name: {m: [] for m in MODELS_TO_RUN}}
    
    for PREDICTION_HORIZON in PREDICTION_HORIZONS:
        # Shift predictor to aligned with the target (y__{t+h} <- x_{t})
        X_transformed_full_shifted = X_transformed_full.shift(PREDICTION_HORIZON)
        X_untransformed_full_shifted = X_untransformed_full.shift(PREDICTION_HORIZON)
        X_yield_full_shifted = X_yield_full.shift(PREDICTION_HORIZON)
        X_ads_full_shifted = X_ads_full.shift(PREDICTION_HORIZON)

        # Main time-series loop
        all_dates = y_target_full.index
        forecast_dates = all_dates[all_dates >= pd.to_datetime(OOS_START_DATE)]
        oos_actuals = y_target_full.loc[forecast_dates, 'USRECM']
        oos_actuals.index.name = 'Date'

        start_time = time.time()

        for i, forecast_date in enumerate(forecast_dates):
            iter_start_time = time.time()

            # --- Prepare training data ---
            train_end_date = forecast_date - pd.DateOffset(months=1)
            y_train_full = y_target_full.loc[:train_end_date, 'USRECM']
            y_actual = oos_actuals.loc[forecast_date]
        
            print(f"Iter {i+1}/{len(forecast_dates)}: h={PREDICTION_HORIZON}, Date={forecast_date.date()}... ", end="")

            # --- Centralized Data Preparation ---
            X_train_transformed_slice = X_transformed_full_shifted.loc[:forecast_date].copy()
            X_untransformed_slice = X_untransformed_full_shifted.loc[:forecast_date].copy()
            X_train_valid_cols = X_train_transformed_slice.drop(columns=X_train_transformed_slice.columns[X_train_transformed_slice.isna().all()]).copy()
            imputer_base = KNNImputer(n_neighbors=5)
            X_train_imputed = pd.DataFrame(imputer_base.fit_transform(X_train_valid_cols), index=X_train_valid_cols.index, columns=X_train_valid_cols.columns)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), index=X_train_imputed.index, columns=X_train_imputed.columns)
        
            # Generate Predictor Sets with current h_qt and q_qt values
            predictor_data_iter = {}

            if pred_set_name == 'TFDI_avg':
                TFDI_dis_raw, all_sub_indices = generate_qTrans_Sub_Indices(X_train_imputed, y_train_full, h_qt=h_qt, q_qt=q_qt)
                predictor_data_iter['TFDI_avg'] = TFDI_dis_raw.mean(axis=1).to_frame('TFDI_avg')
            elif pred_set_name == 'TFDI_pca':
                TFDI_dis_raw, all_sub_indices = generate_qTrans_Sub_Indices(X_train_imputed, y_train_full, h_qt=h_qt, q_qt=q_qt)
                predictor_data_iter['TFDI_pca'] = generate_TFDI_PCA_Factors(TFDI_dis_raw, n_factors=8)
            elif pred_set_name == 'TFDI_Full_pca':
                TFDI_dis_raw, all_sub_indices = generate_qTrans_Sub_Indices(X_train_imputed, y_train_full, h_qt=h_qt, q_qt=q_qt)
                full_8_factors = generate_PCA_Factors(X_train_imputed, n_factors=8)
                TFDI_8_factors = generate_TFDI_PCA_Factors(TFDI_dis_raw, n_factors=8)
                predictor_data_iter['TFDI_Full_pca'] = pd.concat([full_8_factors, TFDI_8_factors], axis=1)

            # Get the predictor data
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

        end_time = time.time()
        print(f"Total computation time: {end_time - start_time:.2f} seconds")

    # Calculate results for this combination
    results_summary = []
    for model_name in MODELS_TO_RUN.keys():
        errors = oos_errors[pred_set_name][model_name]
        valid_errors = [e for e in errors if not np.isnan(e)]
        
        if valid_errors:
            avg_brier = np.mean(valid_errors)
            n_valid = len(valid_errors)
            n_total = len(errors)
            
            results_summary.append({
                'h_qt': h_qt,
                'q_qt': q_qt,
                'Predictor_Set': pred_set_name,
                'Model': model_name,
                'Avg_Brier_Score': avg_brier,
                'Valid_Predictions': n_valid,
                'Total_Predictions': n_total
            })
        else:
            results_summary.append({
                'h_qt': h_qt,
                'q_qt': q_qt,
                'Predictor_Set': pred_set_name,
                'Model': model_name,
                'Avg_Brier_Score': np.nan,
                'Valid_Predictions': 0,
                'Total_Predictions': len(errors)
            })
    
    return results_summary

# %% Main robustness exercise
print("Starting robustness exercise...")
print(f"Testing combinations:")
print(f"h_qt values: {H_QT_VALUES}")
print(f"q_qt values: {Q_QT_VALUES}")
print(f"Total combinations: {len(H_QT_VALUES) * len(Q_QT_VALUES) * len(ALL_POSSIBLE_SETS)}")

all_results = []

# Generate all combinations
combinations = list(itertools.product(H_QT_VALUES, Q_QT_VALUES, ALL_POSSIBLE_SETS))

for i, (h_qt, q_qt, pred_set) in enumerate(combinations):
    print(f"\nProgress: {i+1}/{len(combinations)}")
    
    try:
        results = run_robustness_test(h_qt, q_qt, pred_set)
        all_results.extend(results)
    except Exception as e:
        print(f"Error in combination h_qt={h_qt}, q_qt={q_qt}, pred_set={pred_set}: {e}")
        # Add error entry
        all_results.append({
            'h_qt': h_qt,
            'q_qt': q_qt,
            'Predictor_Set': pred_set,
            'Model': 'Logit',
            'Avg_Brier_Score': np.nan,
            'Valid_Predictions': 0,
            'Total_Predictions': 0,
            'Error': str(e)
        })

# %% Create results DataFrame and save
results_df = pd.DataFrame(all_results)

# Save results
results_filename = f'robustness_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_path = RESULTS_PATH / results_filename
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# %% Display summary results
print("\n" + "="*80)
print("ROBUSTNESS EXERCISE RESULTS")
print("="*80)

# Filter out error entries for analysis
valid_results = results_df[results_df['Avg_Brier_Score'].notna()].copy()

if not valid_results.empty:
    # Best performing combination
    best_combination = valid_results.loc[valid_results['Avg_Brier_Score'].idxmin()]
    print(f"\nBest performing combination:")
    print(f"  h_qt: {best_combination['h_qt']}")
    print(f"  q_qt: {best_combination['q_qt']}")
    print(f"  Predictor Set: {best_combination['Predictor_Set']}")
    print(f"  Brier Score: {best_combination['Avg_Brier_Score']:.6f}")
    
    # Summary by parameter
    print(f"\nSummary by h_qt:")
    for h_qt in H_QT_VALUES:
        subset = valid_results[valid_results['h_qt'] == h_qt]
        if not subset.empty:
            avg_score = subset['Avg_Brier_Score'].mean()
            print(f"  h_qt={h_qt}: Average Brier Score = {avg_score:.6f}")
    
    print(f"\nSummary by q_qt:")
    for q_qt in Q_QT_VALUES:
        subset = valid_results[valid_results['q_qt'] == q_qt]
        if not subset.empty:
            avg_score = subset['Avg_Brier_Score'].mean()
            print(f"  q_qt={q_qt}: Average Brier Score = {avg_score:.6f}")
    
    print(f"\nSummary by Predictor Set:")
    for pred_set in ALL_POSSIBLE_SETS:
        subset = valid_results[valid_results['Predictor_Set'] == pred_set]
        if not subset.empty:
            avg_score = subset['Avg_Brier_Score'].mean()
            print(f"  {pred_set}: Average Brier Score = {avg_score:.6f}")
    
    # Detailed results table
    print(f"\nDetailed Results:")
    print(valid_results.to_string(index=False, float_format='%.6f'))
    
else:
    print("No valid results found.")

print(f"\nRobustness exercise completed. Results saved to: {results_path}") 