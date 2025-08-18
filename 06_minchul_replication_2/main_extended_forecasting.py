# Extended Recursive Forecasting - Deterioration Indices
# Adapted from Colab notebook to VS Code environment
# Based on Rahul's Recursive_Forecasting.ipynb and Minchul's replication

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
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier as HGBoost
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import pointbiserialr
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import brier_score_loss, make_scorer, average_precision_score, log_loss, matthews_corrcoef, f1_score
from sklearn.base import clone
import statsmodels.api as sm
import shap

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

# These are the five models used to generate the ensemble forecasts
MODELS_TO_RUN = {
    'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
    'Logit_L1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
    # 'HGBoost': HGBoost(random_state=42),
    # 'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    # 'RandomForest': RandomForestClassifier(random_state=42),
}

# Rerun Control
FORCE_RERUN_ALL_SETS = True
#FORCE_RERUN_SPECIFIC_SETS = ['Full', 'PCA_Factors_8']  # This is ignored if FORCE_RERUN_ALL_SETS is True
ALL_POSSIBLE_SETS = ['Yield', 'Full', 'PCA_Factors_8', 'ADS', 'Deter_v2', 'Deter_States_v2', 'Deter_PCA_v2', 'Deter_Avg_v2']
#ALL_POSSIBLE_SETS = ['Full', 'PCA_Factors_8']

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

# %% Event study analysis for tcode1 variables
print("\n--- Event Study Analysis ---")
tcode1_vars = ['TB6SMFFM', 'VIXCLSx', 'T1YFFM', 'TB3SMFFM', 'COMPAPFFx',
               'CES0600000007', 'T5YFFM', 'T10YFFM', 'AWHMAN', 'AAAFFM', 'BAAFFM']

data = X_untransformed_full
y_target = y_target_full

recession_starts = y_target[y_target['USRECM'].diff() == 1].index

window_before = 12
window_after = 6
all_events = []

for start_date in recession_starts:
    try:
        start_loc = data.index.get_loc(start_date)

        if start_loc >= window_before:
            event_window = data.iloc[start_loc - window_before : start_loc + window_after + 1]

            if event_window.empty:
                continue

            normalized_window = event_window - event_window.iloc[0]
            normalized_window.index = np.arange(-window_before, window_after + 1)
            all_events.append(normalized_window)

    except KeyError:
        print(f"Recession start date {start_date} not found in data index. Skipping.")
        continue

if not all_events:
    print("Could not generate event study plot: No recessions with a full historical window were found in the sample.")
else:
    # Average the trajectories
    average_event_trajectory = pd.concat(all_events).groupby(level=0).mean()

    fig, ax = plt.subplots(figsize=(14, 9))

    special_vars = ['VIXCLSx', 'AAAFFM', 'BAAFFM']
    other_vars = [v for v in tcode1_vars if v not in special_vars and v in average_event_trajectory.columns]

    if special_vars:
        available_special = [v for v in special_vars if v in average_event_trajectory.columns]
        if available_special:
            average_event_trajectory[available_special].plot(ax=ax, linewidth=3)
    
    if other_vars:
        average_event_trajectory[other_vars].plot(ax=ax, style='--', color='grey', linewidth=1.5, legend=False)

    ax.axvline(x=0, color='red', linestyle='--', label='Recession Start (T=0)')
    ax.set_title('Average Behavior of Stationary Variables Around Recessions', fontsize=20)
    ax.set_xlabel('Months Relative to Recession Start', fontsize=12)
    ax.set_ylabel('Change from T-12 Months (Levels)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'event_study_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% Spread variables analysis
print("\n--- Spread Variables Analysis ---")
spread_vars = ['AAAFFM', 'BAAFFM', 'TB6SMFFM', 'T1YFFM', 'TB3SMFFM', 'COMPAPFFx', 'T5YFFM', 'T10YFFM']

in_sample_spreads = X_untransformed_full.loc[:'1989-12-31', spread_vars]
real_economy_indicator = np.log(X_untransformed_full['INDPRO']).diff().loc[:'1989-12-31']

results = []
for var in spread_vars:
    if var not in in_sample_spreads.columns:
        continue
        
    level = in_sample_spreads[var]
    change = in_sample_spreads[var].diff()

    corr_level = pd.concat([real_economy_indicator, level], axis=1).dropna().corr().iloc[0, 1]
    corr_change = pd.concat([real_economy_indicator, change], axis=1).dropna().corr().iloc[0, 1]

    # A high ratio means the "change" is more informative about the real economy
    if abs(corr_level) > 0:
        ratio = abs(corr_change) / abs(corr_level)
    else:
        ratio = np.nan

    results.append({'Variable': var, 'Corr_Level': corr_level, 'Corr_Change': corr_change, 'Ratio': ratio})

results_df = pd.DataFrame(results).set_index('Variable')
print("Contagion Correlation Ratio (Change vs. Level with INDPRO Growth)")
print(results_df.sort_values(by='Ratio', ascending=False))

# %% Variable groups definition
variable_groups = {
    'Output_Income': ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS', 'CUMFNS'],
    'Labor_Market': ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 'CES2000000008', 'CES3000000008'],
    'Housing': ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS', 'PERMITW'],
    'Consumption_Orders_Inventories': ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'ANDENOx', 'BUSINVx', 'ISRATIOx'],
    'Money_Credit': ['M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST'],
    'Interest_Rates_Spreads': ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx'],
    'Prices': ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'PPICMM', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'],
    'Stock_Market': ['S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx']
}

# %% Main Forecasting Loop
print("\n--- Starting recursive out-of-sample forecasting loop ---")

oos_deter_indices_history = []

# Master Loop for All Horizons
for PREDICTION_HORIZON in PREDICTION_HORIZONS:
    print(f"\n{'='*25} Processing Horizon h={PREDICTION_HORIZON} {'='*25}")

    # Shift predictors to align with target
    X_transformed_shifted = X_transformed_full.shift(PREDICTION_HORIZON)
    X_untransformed_shifted = X_untransformed_full.shift(PREDICTION_HORIZON)
    X_yield_shifted = X_yield_full.shift(PREDICTION_HORIZON)
    X_ads_shifted = X_ads_full.shift(PREDICTION_HORIZON)

    file_path = OOS_PRED_PATH / f'oos_results_h{PREDICTION_HORIZON}_hwindow.pkl'

    horizon_sub_index_path = SUB_INDICES_PATH / f'h{PREDICTION_HORIZON}'
    horizon_sub_index_path.mkdir(exist_ok=True)

    if SAVE_MODELS:
        horizon_model_path = OOS_MODELS_PATH / f'h{PREDICTION_HORIZON}'
        horizon_model_path.mkdir(exist_ok=True)
        print(f"Models for h = {PREDICTION_HORIZON} will be saved to: {horizon_model_path}")

    # Load existing results or initialize new ones
    oos_probs, oos_errors, oos_actuals, oos_importances = {}, {}, None, {}  # Start with empty dicts
    if not FORCE_RERUN_ALL_SETS:
        try:
            print(f"Attempting to load existing results from: {file_path}")
            oos_results = joblib.load(file_path)
            oos_probs = oos_results.get('probabilities', {})  # Use .get for safety
            oos_errors = oos_results.get('squared_errors', {})
            oos_actuals = oos_results.get('actuals', None)
            oos_importances = oos_results.get('importances', {})
            print("Successfully loaded existing results.")
        except FileNotFoundError:
            print("No existing results file found. Initializing new structure.")
    else:
        print(f"FORCE_RERUN_ALL_SETS is True. Any loaded results will be ignored for this horizon.")

    # Determine which predictor sets need to be run
    # ALL_POSSIBLE_SETS = ['Yield', 'Full', 'PCA_Factors_8', 'ADS', 'Deter', 'Deter_States', 'Deter_PCA', 'Deter_Avg']

    sets_to_run = []
    if FORCE_RERUN_ALL_SETS:
        sets_to_run = ALL_POSSIBLE_SETS
        print(f"All {len(sets_to_run)} predictor sets will be re-run.")
    else:
        for pred_set in ALL_POSSIBLE_SETS:
            # Rerun if not found or if specifically requested
            if pred_set not in oos_probs or pred_set in FORCE_RERUN_SPECIFIC_SETS:
                sets_to_run.append(pred_set)
        if FORCE_RERUN_SPECIFIC_SETS:
             print(f"Will force rerun for specific sets: {FORCE_RERUN_SPECIFIC_SETS}")

    if not sets_to_run:
        print("All specified predictor sets have already been run for this horizon. Skipping.")
        continue

    print(f"The following {len(sets_to_run)} sets will be run: {sets_to_run}")

    # Initialize/clear storage ONLY for the sets that are being run
    for pred_set in sets_to_run:
        oos_probs[pred_set] = {m: [] for m in MODELS_TO_RUN}
        oos_errors[pred_set] = {m: [] for m in MODELS_TO_RUN}
        oos_importances[pred_set] = {m: [] for m in MODELS_TO_RUN}

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
                continue
            else:
                X_train_lagged = add_lags(X_train_raw, LAGS_TO_ADD, prefix=f'{pred_set_name}_')

            X_train_lagged_final = X_train_lagged.dropna()
            X_train_final = X_train_lagged_final.loc[:train_end_date]
            X_predict_point = X_train_lagged_final.loc[[forecast_date]]

            for model_name, model_template in MODELS_TO_RUN.items():
                prob, error, importances = np.nan, np.nan, pd.Series()

                try:
                    common_index = y_train_full.index.intersection(X_train_final.index)
                    y_train_aligned = y_train_full.loc[common_index]
                    X_train_aligned = X_train_final.loc[common_index]

                    if len(X_train_aligned) > max(LAGS_TO_ADD, default=0) + 20:
                        model_instance = clone(model_template)

                        if 'XGBoost' in model_name:
                            neg, pos = (y_train_aligned == 0).sum(), (y_train_aligned == 1).sum()
                            if pos > 0: 
                                model_instance = model_template.set_params(scale_pos_weight=(neg/pos))
                        elif 'HGBoost' in model_name or 'RandomForest' in model_name or 'Logit' in model_name:
                            #model_instance = model_template.set_params(class_weight='balanced')
                            model_instance = model_template.set_params(class_weight=None)

                        X_predict_imputed = X_predict_point.reindex(columns=X_train_aligned.columns).ffill().bfill()

                        if not X_predict_imputed.isna().any().any():
                            model_instance.fit(X_train_aligned, y_train_aligned)

                            try:
                                if 'XGBoost' in model_name:
                                    booster = model_instance.get_booster()
                                    importance_dict = booster.get_score(importance_type='gain')
                                    importances = pd.Series(importance_dict, index=X_train_aligned.columns).fillna(0)
                                elif 'Logit' in model_name:
                                    importances = pd.Series(np.abs(model_instance.coef_[0]), index=X_train_aligned.columns)
                                else: # RandomForest, HGBoost
                                    importances = pd.Series(model_instance.feature_importances_, index=X_train_aligned.columns)
                            except Exception as e_imp:
                                print(f"Could not get importances for {model_name} on {forecast_date.date()}: {e_imp}")
                                pass

                            if SAVE_MODELS and forecast_date == forecast_dates[-1]:
                                print("Last iteration. Saving model...")
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

        iter_end_time = time.time()
        print(f" ({(iter_end_time - iter_start_time):.2f}s)")

    # Save results after each horizon's loop is complete
    print(f"\n--- Loop for h={PREDICTION_HORIZON} Finished ---")
    results_to_save = {'probabilities': oos_probs, 'squared_errors': oos_errors, 'actuals': oos_actuals, 'importances': oos_importances}
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

# %% Analysis and Visualization
if 'avg_weights' in locals():
    # Weights for each variable, averaged over sample
    avg_weights_grouped = avg_weights.groupby(level=0).mean()
    top_20_vars = avg_weights_grouped.sort_values(ascending=False).head(20)

    # Create the bar plot
    plt.figure(figsize=(10, 12))
    top_20_vars.sort_values().plot(kind='barh')
    plt.title('Top 20 Most Important Variables (Average Weight)', fontsize=16)
    plt.xlabel('Average Dynamic Stability Weight')
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'top_variables_weights.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% Feature importance analysis
HORIZON_TO_ANALYZE = 12
PREDICTOR_SET_NAME = 'Deter_States_v2'
MODEL_NAME = 'XGBoost'

final_model_filename = f'{PREDICTOR_SET_NAME}_{MODEL_NAME}.pkl'
final_model_path = OOS_MODELS_PATH / f'h{HORIZON_TO_ANALYZE}' / final_model_filename

try:
    final_model = joblib.load(final_model_path)
    print(f"Successfully loaded final model from: {final_model_path}")
    
    y_full = y_target_full['USRECM']
    X_full = X_transformed_full

    feature_names = final_model.feature_names_in_

    if hasattr(final_model, 'coef_'):
        importances = final_model.coef_[0]
        importance_series = pd.Series(abs(importances), index=feature_names)
        print("Model type: Linear (e.g., Logit). Using absolute coefficients.")
    else:
        importances = final_model.feature_importances_
        importance_series = pd.Series(importances, index=feature_names)
        print("Model type: Tree-based. Using feature importances.")

    top_15_vars = importance_series.sort_values(ascending=False).head(15)
    top_var_name = top_15_vars.index[0]

    print(f"\n--- Top 15 Most Important Variables for h={HORIZON_TO_ANALYZE} ---")
    print(top_15_vars)
    print(f"\nThe single most important variable is: '{top_var_name}'")

    # Generate full historical states for visualization
    _, _, full_historical_states = generate_Deter_Indices_v2(X_full, y_full, horizon=HORIZON_TO_ANALYZE)

    if top_var_name in full_historical_states.columns:
        top_signal_series = full_historical_states[top_var_name]
        y_shifted_for_plot = y_full.shift(-HORIZON_TO_ANALYZE)

        fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

        # Plot 1: The Binary Deterioration State
        axes[0].plot(top_signal_series.index, top_signal_series,
                     label=f"Deter_State for {top_var_name}", color='darkred', drawstyle='steps-post')
        axes[0].set_title(f'Behavior of Top Signal for h={HORIZON_TO_ANALYZE}: {top_var_name}', fontsize=16)
        axes[0].set_ylabel('State (1 = Deteriorating)')
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].legend(loc='upper left')

        # Plot 2: The Shifted Recession Bars
        axes[1].fill_between(y_full.index, 0, y_full,
                             where=y_full==1, color='gray', alpha=0.5, label='Actual Recession Dates')
        axes[1].fill_between(y_shifted_for_plot.index, 0, y_shifted_for_plot,
                             where=y_shifted_for_plot==1, color='red', alpha=0.8, label=f'Recession Dates (Shifted Left by {HORIZON_TO_ANALYZE}mo)')
        axes[1].set_title(f'US Recessions (NBER)', fontsize=16)
        axes[1].set_ylabel('Recession (1 = Yes)')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend(loc='upper left')

        axes[1].xaxis.set_major_locator(mdates.YearLocator(5))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / f'top_signal_analysis_h{HORIZON_TO_ANALYZE}.png', dpi=300, bbox_inches='tight')
        plt.show()

except FileNotFoundError:
    print(f"ERROR: Model file not found. Please run the main loop for h={HORIZON_TO_ANALYZE} first.")

print("\n--- Analysis Complete ---")
