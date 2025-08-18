# Factor robustness exercise for TFDI variables

# %% Importing libraries
import os
import pandas as pd
import numpy as np
import time
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
PREDICTION_HORIZONS = [3]
LAGS_TO_ADD = []

# Fixed parameters for TFDI
H_QT_FIXED = 3
Q_QT_FIXED = 0.25

# Factor robustness parameters
N_FACTORS_VALUES = list(range(1, 13))  # 1 to 12

# These are the models used to generate the ensemble forecasts
MODELS_TO_RUN = {
    'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
}

# Predictor set to test
PREDICTOR_SET = 'TFDI_pca'

# %% Load data
y_target_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'y_target.pkl'))
X_yield_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_yield.pkl'))
X_transformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_transformed_monthly.pkl'))
X_untransformed_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_untransformed_monthly.pkl'))
X_ads_full = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_ads.pkl'))
tcodes = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'tcodes.pkl'))

# %% Function to run single factor robustness test
def run_factor_robustness_test(n_factors):
    """
    Run a single factor robustness test with specified number of factors
    """
    print(f"\n{'='*60}")
    print(f"Running factor robustness test: n_factors={n_factors}")
    print(f"Fixed parameters: h_qt={H_QT_FIXED}, q_qt={Q_QT_FIXED}")
    print(f"{'='*60}")
    
    # Initialize storage for results
    oos_probs = {PREDICTOR_SET: {m: [] for m in MODELS_TO_RUN}}
    oos_errors = {PREDICTOR_SET: {m: [] for m in MODELS_TO_RUN}}
    
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
        
            # Generate Predictor Sets with current n_factors
            predictor_data_iter = {}

            # Generate TFDI with fixed h_qt and q_qt, but variable n_factors
            TFDI_dis_raw, all_sub_indices = generate_qTrans_Sub_Indices(X_train_imputed, y_train_full, h_qt=H_QT_FIXED, q_qt=Q_QT_FIXED)
            predictor_data_iter['TFDI_pca'] = generate_TFDI_PCA_Factors(TFDI_dis_raw, n_factors=n_factors)

            # Get the predictor data
            X_train_raw = predictor_data_iter.get(PREDICTOR_SET)
            if X_train_raw is None or X_train_raw.empty:
                for model_name in MODELS_TO_RUN:
                    oos_probs[PREDICTOR_SET][model_name].append(np.nan)
                    oos_errors[PREDICTOR_SET][model_name].append(np.nan)
                continue
            else:
                X_train_lagged = add_lags(X_train_raw, LAGS_TO_ADD, prefix=f'{PREDICTOR_SET}_')

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
                    print(f"Error in model {model_name} for n_factors={n_factors} at date {forecast_date}: {e}")
            
                oos_probs[PREDICTOR_SET][model_name].append(prob)
                oos_errors[PREDICTOR_SET][model_name].append(error)

            iter_end_time = time.time()
            print(f"  Iteration time: {iter_end_time - iter_start_time:.2f} seconds")

        end_time = time.time()
        print(f"Total computation time: {end_time - start_time:.2f} seconds")

    # Calculate results for this combination
    results_summary = []
    for model_name in MODELS_TO_RUN.keys():
        errors = oos_errors[PREDICTOR_SET][model_name]
        valid_errors = [e for e in errors if not np.isnan(e)]
        
        if valid_errors:
            avg_brier = np.mean(valid_errors)
            n_valid = len(valid_errors)
            n_total = len(errors)
            
            results_summary.append({
                'n_factors': n_factors,
                'h_qt': H_QT_FIXED,
                'q_qt': Q_QT_FIXED,
                'Predictor_Set': PREDICTOR_SET,
                'Model': model_name,
                'Avg_Brier_Score': avg_brier,
                'Valid_Predictions': n_valid,
                'Total_Predictions': n_total
            })
        else:
            results_summary.append({
                'n_factors': n_factors,
                'h_qt': H_QT_FIXED,
                'q_qt': Q_QT_FIXED,
                'Predictor_Set': PREDICTOR_SET,
                'Model': model_name,
                'Avg_Brier_Score': np.nan,
                'Valid_Predictions': 0,
                'Total_Predictions': len(errors)
            })
    
    return results_summary

# %% Main factor robustness exercise
print("Starting factor robustness exercise...")
print(f"Testing n_factors from {min(N_FACTORS_VALUES)} to {max(N_FACTORS_VALUES)}")
print(f"Fixed parameters: h_qt={H_QT_FIXED}, q_qt={Q_QT_FIXED}")
print(f"Predictor set: {PREDICTOR_SET}")
print(f"Total combinations: {len(N_FACTORS_VALUES)}")

all_results = []

for i, n_factors in enumerate(N_FACTORS_VALUES):
    print(f"\nProgress: {i+1}/{len(N_FACTORS_VALUES)}")
    
    try:
        results = run_factor_robustness_test(n_factors)
        all_results.extend(results)
    except Exception as e:
        print(f"Error in n_factors={n_factors}: {e}")
        # Add error entry
        all_results.append({
            'n_factors': n_factors,
            'h_qt': H_QT_FIXED,
            'q_qt': Q_QT_FIXED,
            'Predictor_Set': PREDICTOR_SET,
            'Model': 'Logit',
            'Avg_Brier_Score': np.nan,
            'Valid_Predictions': 0,
            'Total_Predictions': 0,
            'Error': str(e)
        })

# %% Create results DataFrame and save
results_df = pd.DataFrame(all_results)

# Save results
results_filename = f'factor_robustness_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_path = RESULTS_PATH / results_filename
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# %% Display summary results
print("\n" + "="*80)
print("FACTOR ROBUSTNESS EXERCISE RESULTS")
print("="*80)

# Filter out error entries for analysis
valid_results = results_df[results_df['Avg_Brier_Score'].notna()].copy()

if not valid_results.empty:
    # Best performing combination
    best_combination = valid_results.loc[valid_results['Avg_Brier_Score'].idxmin()]
    print(f"\nBest performing combination:")
    print(f"  n_factors: {best_combination['n_factors']}")
    print(f"  Brier Score: {best_combination['Avg_Brier_Score']:.6f}")
    
    # Summary statistics
    print(f"\nSummary statistics:")
    print(f"  Mean Brier Score: {valid_results['Avg_Brier_Score'].mean():.6f}")
    print(f"  Std Brier Score: {valid_results['Avg_Brier_Score'].std():.6f}")
    print(f"  Min Brier Score: {valid_results['Avg_Brier_Score'].min():.6f}")
    print(f"  Max Brier Score: {valid_results['Avg_Brier_Score'].max():.6f}")
    
    # Performance by number of factors
    print(f"\nPerformance by number of factors:")
    for n_factors in sorted(valid_results['n_factors'].unique()):
        subset = valid_results[valid_results['n_factors'] == n_factors]
        if not subset.empty:
            avg_score = subset['Avg_Brier_Score'].iloc[0]  # Should be only one row per n_factors
            print(f"  n_factors={n_factors:2d}: Brier Score = {avg_score:.6f}")
    
    # Find optimal number of factors
    optimal_n_factors = valid_results.loc[valid_results['Avg_Brier_Score'].idxmin(), 'n_factors']
    print(f"\nOptimal number of factors: {optimal_n_factors}")
    
    # Detailed results table
    print(f"\nDetailed Results:")
    print(valid_results[['n_factors', 'Avg_Brier_Score', 'Valid_Predictions', 'Total_Predictions']].to_string(index=False, float_format='%.6f'))
    
else:
    print("No valid results found.")

print(f"\nFactor robustness exercise completed. Results saved to: {results_path}") 