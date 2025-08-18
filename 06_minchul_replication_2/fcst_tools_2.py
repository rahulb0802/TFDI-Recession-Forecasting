# fcst_tools_2.py
# Additional forecasting tools for TFDI analysis

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import pointbiserialr


def learn_in_sample_static_weights(X_in_sample, y_in_sample, variable_groups):
    """
    Learns a single, stable set of weights based on the correlation and
    autocorrelation of signals during a fixed in-sample period.
    This is a supervised method that is free of lookahead bias when the
    weights are applied to a subsequent out-of-sample period.
    """
    print("Learning static, in-sample weights using corr * autocorr...")

    in_sample_states = pd.DataFrame(index=X_in_sample.index)
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'VIXCLSx', 'BAAFFM', 'AAAFFM'}
    special_financial_vars = {'VIXCLSx', 'BAAFFM', 'AAAFFM'}

    for var in X_in_sample.columns:
        if var not in variable_groups.get(get_category_for_var(var, variable_groups), []): 
            continue

        signal_for_ranking = X_in_sample[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        if var in special_financial_vars:
            signal_for_ranking = X_in_sample[var].diff()

        input_signal = signal_for_ranking.rolling(window=3, min_periods=1).mean()

        deterioration_threshold = input_signal.expanding(min_periods=36).quantile(0.75 if use_counter_logic else 0.25)

        deteriorating_state = pd.Series(0.0, index=input_signal.index)
        if use_counter_logic:
            deteriorating_state[input_signal > deterioration_threshold] = 1.0
        else:
            deteriorating_state[input_signal < deterioration_threshold] = 1.0
        in_sample_states[var] = deteriorating_state

    in_sample_states = in_sample_states.fillna(method='ffill').fillna(0)

    static_weights = {}
    for category, var_list in variable_groups.items():
        det_cols = [v for v in var_list if v in in_sample_states.columns]
        if not det_cols:
            continue

        signals_in_cat = in_sample_states[det_cols]

        data_for_corr = pd.concat([y_in_sample, signals_in_cat], axis=1).dropna()
        predictive_power = data_for_corr.corr().iloc[0, 1:].abs()

        stability = signals_in_cat.apply(lambda s: s.autocorr(lag=1)).fillna(0).abs()

        static_weights[category] = predictive_power * stability

    category_importance_scores = {cat: weights.mean() for cat, weights in static_weights.items() if not weights.empty}

    print("Finished learning static weights.")
    return static_weights, category_importance_scores


def get_category_for_var(var_name, variable_groups):
    """Helper function to find which category a variable belongs to"""
    for category, var_list in variable_groups.items():
        if var_name in var_list:
            return category
    return None


def select_top_variables_per_category(X_data, y_data, variable_groups, horizon, top_n=10, corr_threshold=0.1):
    """
    Selects the top N variables per category based on correlation.
    """
    print("      -> Selecting top variables per category (Strict Top-N)...")

    y_shifted = y_data.shift(-horizon)
    y_shifted.name = 'y_lead'

    aligned_data = pd.concat([y_shifted, X_data], axis=1).dropna()
    y_aligned = aligned_data['y_lead']
    X_aligned = aligned_data.drop(columns=['y_lead'])

    if len(y_aligned.unique()) < 2:
        return {cat: vars[:top_n] for cat, vars in variable_groups.items()}  # Fallback

    refined_groups = {}
    for category, var_list in variable_groups.items():
        var_scores = {}
        for var in var_list:
            if var in X_aligned.columns:
                try:
                    correlation, _ = pointbiserialr(X_aligned[var], y_aligned)
                    if not np.isnan(correlation) and abs(correlation) >= corr_threshold:
                        var_scores[var] = abs(correlation)
                except ValueError:
                    continue

        # Sort variables by score
        sorted_vars = sorted(var_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract only the names of the top N variables
        top_var_names = [var for var, score in sorted_vars[:top_n]]

        if top_var_names:
            refined_groups[category] = top_var_names  # Store the list of names

    return refined_groups


def get_final_definitive_span(X_transformed_data, variable_groups):
    """
    Calculates the single, definitive, system-wide optimal span based on the
    median persistence of all variables actively used in the index
    """
    print("Calculating the Final, Definitive System-Wide Optimal Span...")

    relevant_vars = [var for var_list in variable_groups.values() for var in var_list]

    X_relevant = X_transformed_data[[v for v in relevant_vars if v in X_transformed_data.columns]]

    # Calculate autocorrelation
    autocorrelations = X_relevant.apply(lambda s: s.autocorr(lag=1)).fillna(0).abs()

    median_persistence = autocorrelations.median()

    optimal_span = 1 / (median_persistence + 1e-9)

    print(f"\n--- Definitive Results ---")
    print(f"Median Persistence of Relevant TRANSFORMED Signals: {median_persistence:.4f}")
    print(f"Calculated System-Wide Optimal Span: {optimal_span:.4f}")

    return optimal_span


def get_horizon_specific_optimal_span(X_transformed_data, y_target_data, horizon, variable_groups):
    """
    Calculates the definitive optimal span for a specific horizon by focusing on the
    persistence of the most predictive variables at that horizon.
    """
    print(f"      -> Calculating Horizon-Specific Optimal Span for h={horizon}...")

    # 1. Identify the most predictive variables for THIS horizon
    y_shifted = y_target_data.shift(-horizon)
    relevant_vars = [var for var_list in variable_groups.values() for var in var_list if var in X_transformed_data.columns]
    X_relevant = X_transformed_data[relevant_vars]

    data_for_corr = pd.concat([y_shifted, X_relevant], axis=1).dropna()
    predictive_power = data_for_corr.corr().iloc[0, 1:].abs()

    # 2. Filter for the "star players" (above the median)
    median_power = predictive_power.median()
    star_players = predictive_power[predictive_power > median_power].index.tolist()

    # 3. Calculate persistence of ONLY these star players
    X_star_players = X_relevant[star_players]
    autocorrelations = X_star_players.apply(lambda s: s.autocorr(lag=1)).fillna(0).abs()

    # 4. Find the robust median persistence of this high-quality group
    median_persistence = autocorrelations.median()

    # 5. Convert this to the final, optimal span for this horizon
    optimal_span = 1 / (median_persistence + 1e-9)

    print(f"         ... Median Persistence of Top Predictors for h={horizon}: {median_persistence:.4f}")
    print(f"         ... Calculated Optimal Span for h={horizon}: {optimal_span:.4f}")

    return optimal_span


def generate_PCA_Factors_Binary(X_transformed_train, n_factors=8):
    """
    Returns the top PCA factors for binary data.
    """
    print("      -> Generating PCA Factors (Binary)...")

    # Drop columns that are entirely NaN in the CURRENT training slice
    X_stat = X_transformed_train.copy()
    cols_to_drop_nan = X_stat.columns[X_stat.isna().all()]
    if not cols_to_drop_nan.empty:
        print(f"\n         ... Dropping {len(cols_to_drop_nan)} all-NaN columns: {cols_to_drop_nan.to_list()}", end="")
    X_stat_valid = X_stat.drop(columns=cols_to_drop_nan)

    X_imputed = X_stat_valid.fillna(0)  # missing value implies state of nondeterioration

    # Drop constant columns AFTER scaling (a final check)
    variances = X_imputed.var()
    constant_cols = variances[variances < 1e-10].index
    if not constant_cols.empty:
        print(f"\n         ... Dropping {len(constant_cols)} constant columns: {constant_cols.to_list()}", end="")
    X_final_for_pca = X_imputed.drop(columns=constant_cols)

    # PCA on the final, clean data
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X_final_for_pca)

    pca_factors_df = pd.DataFrame(factors,
                                  index=X_final_for_pca.index,
                                  columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    return pca_factors_df


def generate_TFDI_Sub_Indices_v2(X_transformed_train, y_train, horizon):
    """
    This is the TDFI framework. It uses Ridge (L2) for nowcasting (h<3)
    to retain all signals, and LASSO (L1) for forecasting (h>=3) to perform
    automated feature selection and remove noise.
    """
    print(f"      -> Generating TFDI v2 (h={horizon})...")

    # Generate Unweighted Sub-Indices
    variable_groups = {
        'Output_Income': ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS', 'CUMFNS'],
        'Labor_Market': ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 'CES2000000008', 'CES3000000008'],
        'Housing': ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS', 'PERMITW'],
        'Consumption_Orders_Inventories': ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'ANDENOx', 'BUSINVx', 'ISRATIOx'],
        'Money_Credit': ['M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST'],
        'Interest_Rates_Spreads': ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM'],
        'FX_Rates': ['EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx'],
        'Prices': ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'PPICMM', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'],
        'Stock_Market': ['S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx']
    }
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    special_financial_vars = {'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    X_momentum = X_transformed_train.rolling(window=3, min_periods=1).mean()
    
    refined_variable_groups = variable_groups

    adaptive_window = 180
    weakness_states = pd.DataFrame(index=X_transformed_train.index)
    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    for var in all_selected_vars:
        if var not in X_transformed_train.columns:
            continue
            
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        if horizon == 1 and var in special_financial_vars:
            use_counter_logic = True
            signal_for_ranking = X_transformed_train[var].diff()
        elif horizon > 1 and var in special_financial_vars:
            signal_for_ranking = X_transformed_train[var].diff()

        level_signal = signal_for_ranking
        momentum_signal = signal_for_ranking.rolling(window=3, min_periods=1).mean()

        weakness_threshold = level_signal.quantile(0.8 if use_counter_logic else 0.2)
        weak_state = pd.Series(0.0, index=level_signal.index)
        if use_counter_logic: 
            weak_state[level_signal > weakness_threshold] = 1.0
        else: 
            weak_state[level_signal < weakness_threshold] = 1.0
        weakness_states[var] = weak_state

        deterioration_threshold = momentum_signal.quantile(0.8 if use_counter_logic else 0.2)
        deteriorating_state = pd.Series(0.0, index=momentum_signal.index)
        if use_counter_logic: 
            deteriorating_state[momentum_signal > deterioration_threshold] = 1.0
        else: 
            deteriorating_state[momentum_signal < deterioration_threshold] = 1.0
        deterioration_states[var] = deteriorating_state

    # Aggregate into Unweighted Per-Category Sub-Indices
    cat_weakness_di = pd.DataFrame(index=X_transformed_train.index)
    cat_deterioration_di = pd.DataFrame(index=X_transformed_train.index)

    for category, var_list in refined_variable_groups.items():
        # Only try to access columns that were actually selected
        weak_cols = [v for v in var_list if v in weakness_states.columns]
        if weak_cols:
            cat_weakness_di[f"W_{category.replace(' ', '_')}"] = weakness_states[weak_cols].mean(axis=1)

        det_cols = [v for v in var_list if v in deterioration_states.columns]
        if det_cols:
            cat_deterioration_di[f"D_{category.replace(' ', '_')}"] = deterioration_states[det_cols].mean(axis=1)

    all_sub_indices = pd.concat([cat_weakness_di, cat_deterioration_di], axis=1)
    
    # Weighting step
    y_shifted = y_train.shift(-horizon).rename('y_lead')
    weighting_data = pd.concat([y_shifted, all_sub_indices], axis=1, join='inner').dropna()
    y_weight = weighting_data['y_lead']
    X_weight = weighting_data.drop(columns=['y_lead'])

    weights = pd.Series(1.0, index=X_weight.columns)  # Default to equal weights

    if len(y_weight.unique()) == 2 and not X_weight.empty:
        if horizon < 3:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
        else:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)

        # Not fitting gridsearch because tuning does worse
        model.fit(X_weight, y_weight)
        weights = pd.Series(np.abs(model.coef_[0]), index=X_weight.columns)

    return all_sub_indices.mul(weights).fillna(method='ffill').fillna(0)


def generate_Weakness_Indices(X_transformed_train, y_train, horizon):
    """
    Generate weakness indices only.
    """
    print(f"      -> Generating TFDI Weakness Indices (h={horizon})...")

    variable_groups = {
        'Output_Income': ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS', 'CUMFNS'],
        'Labor_Market': ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 'CES2000000008', 'CES3000000008'],
        'Housing': ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS', 'PERMITW'],
        'Consumption_Orders_Inventories': ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'BUSINVx', 'ISRATIOx'],
        'Money_Credit': ['M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST'],
        'Interest_Rates_Spreads': ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM'],
        'FX_Rates': ['EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx'],
        'Prices': ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'PPICMM', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'],
        'Stock_Market': ['S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx']
    }
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    special_financial_vars = {'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    
    refined_variable_groups = variable_groups
    adaptive_window = 180

    weakness_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    for var in all_selected_vars:
        if var not in X_transformed_train.columns:
            continue
            
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        if horizon == 1 and var in special_financial_vars:
            use_counter_logic = True
            signal_for_ranking = X_transformed_train[var].diff()
        elif horizon > 1 and var in special_financial_vars:
            signal_for_ranking = X_transformed_train[var].diff()

        level_signal = signal_for_ranking

        weakness_threshold = level_signal.rolling(window=adaptive_window, min_periods=36).quantile(0.8 if use_counter_logic else 0.2)
        weak_state = pd.Series(0.0, index=level_signal.index)
        if use_counter_logic: 
            weak_state[level_signal > weakness_threshold] = 1.0
        else: 
            weak_state[level_signal < weakness_threshold] = 1.0
        weakness_states[var] = weak_state

    # Perform same operations as original function but only for Weakness
    cat_weakness_di = pd.DataFrame(index=X_transformed_train.index)

    for category, var_list in refined_variable_groups.items():
        weak_cols = [v for v in var_list if v in weakness_states.columns]
        if weak_cols:
            cat_weakness_di[f"W_{category.replace(' ', '_')}"] = weakness_states[weak_cols].mean(axis=1)

    all_sub_indices = cat_weakness_di

    y_shifted = y_train.shift(-horizon).rename('y_lead')
    weighting_data = pd.concat([y_shifted, all_sub_indices], axis=1, join='inner').dropna()
    y_weight = weighting_data['y_lead']
    X_weight = weighting_data.drop(columns=['y_lead'])

    weights = pd.Series(1.0, index=X_weight.columns)  # Default to equal weights

    if len(y_weight.unique()) == 2 and not X_weight.empty:
        if horizon < 3:
            print("         ... using LASSO (L2) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
        else:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)

        # Not fitting gridsearch because tuning does worse
        model.fit(X_weight, y_weight)
        weights = pd.Series(np.abs(model.coef_[0]), index=X_weight.columns)

    return all_sub_indices.mul(weights).fillna(method='ffill').fillna(0)


def generate_Deter_Indices(X_transformed_train, y_train, horizon, window=3):
    """
    Generate deterioration indices.
    """
    print(f"      -> Generating Deterioration Indices (h={horizon})...")

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
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'VIXCLSx', 'BAAFFM', 'AAAFFM'}
    special_financial_vars = {'VIXCLSx', 'BAAFFM', 'AAAFFM'}

    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in variable_groups.values() for var in var_list]
    for var in all_selected_vars:
        if var not in X_transformed_train.columns:
            continue
            
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        volatility = signal_for_ranking.std()

        momentum = signal_for_ranking.rolling(window=horizon, min_periods=1).mean()
        input_signal = momentum

        quantile = 0.25
        counter_quantile = 1 - quantile

        deterioration_threshold = input_signal.quantile(counter_quantile if use_counter_logic else quantile)
        deteriorating_state = pd.Series(0.0, index=input_signal.index)
        if use_counter_logic: 
            deteriorating_state[input_signal > deterioration_threshold] = 1.0
        else: 
            deteriorating_state[input_signal < deterioration_threshold] = 1.0
        deterioration_states[var] = deteriorating_state

    cat_deterioration_di = pd.DataFrame(index=X_transformed_train.index)

    all_weights = {}

    for category, var_list in variable_groups.items():
        det_cols = [v for v in var_list if v in deterioration_states.columns]
        signals_in_cat = deterioration_states[det_cols]

        # Weights are determined by corr * autocorr
        data_for_corr = pd.concat([y_train, signals_in_cat], axis=1).dropna()

        if not data_for_corr.empty and data_for_corr.iloc[:, 0].nunique() > 1:
            predictive_power = data_for_corr.corr().iloc[0, 1:].abs()
        else:
            predictive_power = pd.Series(1.0, index=signals_in_cat.columns)

        stability = signals_in_cat.apply(lambda s: s.autocorr(lag=1)).fillna(0).abs()

        all_weights[category] = predictive_power * stability

    # Overall weight is average variable weight within cat
    category_importance_scores = {cat: weights.mean() for cat, weights in all_weights.items()}

    all_weights_for_analysis = []

    for category, var_list in variable_groups.items():
        det_cols = [v for v in var_list if v in deterioration_states.columns]

        signals_in_cat = deterioration_states[det_cols]
        weights_in_cat = all_weights.get(category)

        all_weights_for_analysis.append(weights_in_cat)

        weighted_signals = signals_in_cat * weights_in_cat
        proportional_breadth_index = weighted_signals.mean(axis=1)

        category_amplifier = category_importance_scores.get(category, 1.0)

        # No amplifier
        final_index = proportional_breadth_index

        cat_deterioration_di[f"D_{category.replace(' ', '_')}"] = final_index

    all_sub_indices = cat_deterioration_di.ffill().fillna(0)

    final_weights_df = pd.concat(all_weights_for_analysis)

    return all_sub_indices, final_weights_df, deterioration_states


def generate_Deter_Avg_Indices(X_transformed_train, y_train, horizon, window=3):
    """
    Generate average deterioration indices.
    """
    print(f"      -> Generating Average Deterioration Indices (h={horizon})...")

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
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'VIXCLSx', 'BAAFFM', 'AAAFFM'}
    special_financial_vars = {'VIXCLSx', 'BAAFFM', 'AAAFFM'}

    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in variable_groups.values() for var in var_list]
    for var in all_selected_vars:
        if var not in X_transformed_train.columns:
            continue
            
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        volatility = signal_for_ranking.std()

        momentum = signal_for_ranking.rolling(window=horizon, min_periods=1).median()
        input_signal = momentum

        deterioration_threshold = input_signal.quantile(0.75 if use_counter_logic else 0.25)
        deteriorating_state = pd.Series(0.0, index=input_signal.index)
        if use_counter_logic: 
            deteriorating_state[input_signal > deterioration_threshold] = 1.0
        else: 
            deteriorating_state[input_signal < deterioration_threshold] = 1.0
        deterioration_states[var] = deteriorating_state

    cat_deterioration_di = pd.DataFrame(index=X_transformed_train.index)

    for category, var_list in variable_groups.items():
        det_cols = [v for v in var_list if v in deterioration_states.columns]
        if det_cols:
            cat_deterioration_di[f"D_{category.replace(' ', '_')}"] = deterioration_states[det_cols].mean(axis=1)

    all_sub_indices = cat_deterioration_di.ffill().fillna(0)

    return all_sub_indices, deterioration_states


def add_lags_wo_current(df, lags_to_add, prefix=''):
    """
    Adds lagged versions of columns to a DataFrame without the current period.
    """
    if not lags_to_add:
        return df

    df_lagged_list = []
    for lag in lags_to_add:
        df_shifted = df.shift(lag)
        df_shifted.columns = [f'{prefix}{col}_lag{lag}' for col in df.columns]
        df_lagged_list.append(df_shifted)

    if df_lagged_list:
        return pd.concat(df_lagged_list, axis=1)
    else:
        return pd.DataFrame(index=df.index)


def add_lags(df, lags_to_add, prefix=''):
    """
    Adds lagged versions of columns to a DataFrame.
    """
    if not lags_to_add:
        return df

    df_lagged = df.copy()
    for lag in lags_to_add:
        df_shifted = df.shift(lag)
        df_shifted.columns = [f'{prefix}{col}_lag{lag}' for col in df.columns]
        df_lagged = pd.concat([df_lagged, df_shifted], axis=1)

    return df_lagged
