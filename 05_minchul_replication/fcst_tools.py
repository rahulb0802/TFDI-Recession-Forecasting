# Forecasting Tools Module
# Contains utility functions for recursive forecasting

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from scipy.stats import pointbiserialr

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
        return {cat: vars[:top_n] for cat, vars in variable_groups.items()} # Fallback

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
            refined_groups[category] = top_var_names # Store the list of names

    return refined_groups



def generate_PCA_Factors(X_transformed_train, n_factors=8):
    """
    Returns the top PCA factors.
    """
    print("      -> Generating PCA Factors...")

    # Drop columns that are entirely NaN in the CURRENT training slice
    X_stat = X_transformed_train.copy()
    cols_to_drop_nan = X_stat.columns[X_stat.isna().all()]
    if not cols_to_drop_nan.empty:
        print(f"\n         ... Dropping {len(cols_to_drop_nan)} all-NaN columns: {cols_to_drop_nan.to_list()}", end="")
    X_stat_valid = X_stat.drop(columns=cols_to_drop_nan)

    # Outlier Treatment on the valid data
    for col in X_stat_valid.columns:
        mean, std = X_stat_valid[col].mean(), X_stat_valid[col].std()
        if std > 0: # Avoid dividing by zero for constant columns
            upper, lower = mean + 5 * std, mean - 5 * std
            X_stat_valid[col] = X_stat_valid[col].clip(lower=lower, upper=upper)

    # Imputation. Now guaranteed to have matching shapes
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_stat_valid),
                             index=X_stat_valid.index,
                             columns=X_stat_valid.columns)

    # Standardization
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed),
                            index=X_imputed.index,
                            columns=X_imputed.columns)

    # Drop constant columns AFTER scaling (a final check)
    variances = X_scaled.var()
    constant_cols = variances[variances < 1e-10].index
    if not constant_cols.empty:
        print(f"\n         ... Dropping {len(constant_cols)} constant columns: {constant_cols.to_list()}", end="")
    X_final_for_pca = X_scaled.drop(columns=constant_cols)


    # PCA on the final, clean data
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X_final_for_pca)

    pca_factors_df = pd.DataFrame(factors,
                                  index=X_final_for_pca.index,
                                  columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    # Sign Standardization
    if 'INDPRO' in X_final_for_pca.columns and pca_factors_df['PCA_Factor_1'].corr(X_final_for_pca['INDPRO']) < 0.02:
        pca_factors_df['PCA_Factor_1'] = -pca_factors_df['PCA_Factor_1']

    # Yield spread proxy for Factor 2
    spread_proxy = None
    if 'T10Y3M' in X_final_for_pca.columns:
        spread_proxy = 'T10Y3M'
    elif 'T10Y2Y' in X_final_for_pca.columns:
        spread_proxy = 'T10Y2Y'

    if spread_proxy and pca_factors_df['PCA_Factor_2'].corr(X_final_for_pca[spread_proxy]) > 0.02:
        pca_factors_df['PCA_Factor_2'] = -pca_factors_df['PCA_Factor_2']

    return pca_factors_df



def generate_TFDI_Sub_Indices(X_transformed_train, y_train, horizon):
    """
    This is the TDFI framework. It uses Ridge (L2) for nowcasting (h<3)
    to retain all signals, and LASSO (L1) for forecasting (h>=3) to perform
    automated feature selection and remove noise.
    """
    print(f"      -> Generating TFDI (h={horizon})...")

    # Generate Unweighted Sub-Indices
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
    X_momentum = X_transformed_train.rolling(window=3, min_periods=1).mean()
    refined_variable_groups = select_top_variables_per_category(X_momentum, y_train, variable_groups, horizon=horizon, top_n=10)
    weakness_states = pd.DataFrame(index=X_transformed_train.index)
    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    for var in all_selected_vars:
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


        weakness_threshold = level_signal.quantile(0.75 if use_counter_logic else 0.25)
        weak_state = pd.Series(0.0, index=level_signal.index)
        if use_counter_logic: weak_state[level_signal > weakness_threshold] = 1.0
        else: weak_state[level_signal < weakness_threshold] = 1.0
        weakness_states[var] = weak_state

        deterioration_threshold = momentum_signal.quantile(0.75 if use_counter_logic else 0.25)
        deteriorating_state = pd.Series(0.0, index=momentum_signal.index)
        if use_counter_logic: deteriorating_state[momentum_signal > deterioration_threshold] = 1.0
        else: deteriorating_state[momentum_signal < deterioration_threshold] = 1.0
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

    weights = pd.Series(1.0, index=X_weight.columns) # Default to equal weights

    if len(y_weight.unique()) == 2 and not X_weight.empty:


        if horizon < 3:
            print("         ... using Ridge (L2) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
        else:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'C': [1.0, 0.5, 0.1, 0.05, 0.01]}

        # Not fitting gridsearch because tuning does worse
        model.fit(X_weight, y_weight)
        weights = pd.Series(np.abs(model.coef_[0]), index=X_weight.columns)

    return all_sub_indices.mul(weights).fillna(method='ffill').fillna(0), all_sub_indices.fillna(method='ffill').fillna(0)

def generate_Weakness_Indices(X_transformed_train, y_train, horizon):
    """
    This is the final, unified framework. It uses Ridge (L2) for nowcasting (h<3)
    to retain all signals, and LASSO (L1) for forecasting (h>=3) to perform
    automated feature selection and remove noise.
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
    X_momentum = X_transformed_train.rolling(window=3, min_periods=1).mean()
    refined_variable_groups = select_top_variables_per_category(X_momentum, y_train, variable_groups, horizon=horizon, top_n=10)
    weakness_states = pd.DataFrame(index=X_transformed_train.index)
    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    for var in all_selected_vars:
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


        weakness_threshold = level_signal.quantile(0.75 if use_counter_logic else 0.25)
        weak_state = pd.Series(0.0, index=level_signal.index)
        if use_counter_logic: weak_state[level_signal > weakness_threshold] = 1.0
        else: weak_state[level_signal < weakness_threshold] = 1.0
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

    weights = pd.Series(1.0, index=X_weight.columns) # Default to equal weights

    if len(y_weight.unique()) == 2 and not X_weight.empty:



        if horizon < 3:
            print("         ... using Ridge (L2) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
        else:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'C': [1.0, 0.5, 0.1, 0.05, 0.01]}

        # Not fitting gridsearch because tuning does worse
        model.fit(X_weight, y_weight)
        weights = pd.Series(np.abs(model.coef_[0]), index=X_weight.columns)


    return all_sub_indices.mul(weights).fillna(method='ffill').fillna(0), all_sub_indices.fillna(method='ffill').fillna(0)

def generate_Deter_Indices(X_transformed_train, y_train, horizon):
    """
    This is the final, unified framework. It uses Ridge (L2) for nowcasting (h<3)
    to retain all signals, and LASSO (L1) for forecasting (h>=3) to perform
    automated feature selection and remove noise.
    """
    print(f"      -> Generating Deterioration Indices (h={horizon})...")

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
    X_momentum = X_transformed_train.rolling(window=3, min_periods=1).mean()
    refined_variable_groups = select_top_variables_per_category(X_momentum, y_train, variable_groups, horizon=horizon, top_n=10)
    weakness_states = pd.DataFrame(index=X_transformed_train.index)
    deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    for var in all_selected_vars:
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        # Use transformed values for special financial vars, not levels
        if horizon == 0 and var in special_financial_vars:
            use_counter_logic = True
            signal_for_ranking = X_transformed_train[var].diff()
        elif horizon > 0 and var in special_financial_vars:
            signal_for_ranking = X_transformed_train[var].diff()

        level_signal = signal_for_ranking
        momentum_signal = signal_for_ranking.rolling(window=3, min_periods=1).mean()


        deterioration_threshold = momentum_signal.quantile(0.75 if use_counter_logic else 0.25)
        deteriorating_state = pd.Series(0.0, index=momentum_signal.index)
        if use_counter_logic: deteriorating_state[momentum_signal > deterioration_threshold] = 1.0
        else: deteriorating_state[momentum_signal < deterioration_threshold] = 1.0
        deterioration_states[var] = deteriorating_state


    # Aggregate into Unweighted Per-Category Sub-Indices
    cat_deterioration_di = pd.DataFrame(index=X_transformed_train.index)

    for category, var_list in refined_variable_groups.items():

        det_cols = [v for v in var_list if v in deterioration_states.columns]
        if det_cols:
            cat_deterioration_di[f"D_{category.replace(' ', '_')}"] = deterioration_states[det_cols].mean(axis=1)


    all_sub_indices = cat_deterioration_di
    # Weighting stage
    y_shifted = y_train.shift(-horizon).rename('y_lead')
    weighting_data = pd.concat([y_shifted, all_sub_indices], axis=1, join='inner').dropna()
    y_weight = weighting_data['y_lead']
    X_weight = weighting_data.drop(columns=['y_lead'])

    weights = pd.Series(1.0, index=X_weight.columns) # Default to equal weights

    if len(y_weight.unique()) == 2 and not X_weight.empty:



        if horizon < 3:
            print("         ... using Ridge (L2) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
        else:
            print("         ... using LASSO (L1) weighting for forecasting.")
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1)
            param_grid = {'C': [1.0, 0.5, 0.1, 0.05, 0.01]}

        # Not fitting gridsearch because tuning does worse
        model.fit(X_weight, y_weight)
        weights = pd.Series(np.abs(model.coef_[0]), index=X_weight.columns)


    return all_sub_indices.mul(weights).fillna(method='ffill').fillna(0), all_sub_indices.fillna(method='ffill').fillna(0)


def add_lags(df, lags_to_add, prefix=''):
    """
    Adds lagged versions of columns to a DataFrame.
    """
    if not lags_to_add:
        return df

    # Start with the original DataFrame
    df_list = [df.copy()]
    
    # Add lagged versions
    for lag in lags_to_add:
        df_shifted = df.shift(lag)
        df_shifted.columns = [f'{prefix}{col}_lag{lag}' for col in df.columns]
        df_list.append(df_shifted)

    # Concatenate all DataFrames at once to avoid fragmentation
    return pd.concat(df_list, axis=1)


def add_lags_wo_current(df, lags_to_add, prefix=''):
    """
    Adds lagged versions of columns to a DataFrame.
    Only includes the current df if lags_to_add includes 0.
    """
    if not lags_to_add:
        return df

    # Start with an empty list of DataFrames to concatenate
    df_list = []
    
    # Only include current df if 0 is in lags_to_add
    if 0 in lags_to_add:
        df_current = df.copy()
        df_current.columns = [f'{prefix}{col}_lag0' for col in df.columns]
        df_list.append(df_current)
    
    # Add lagged versions for non-zero lags
    for lag in lags_to_add:
        if lag != 0:  # Skip lag 0 as it's handled above
            df_shifted = df.shift(lag)
            df_shifted.columns = [f'{prefix}{col}_lag{lag}' for col in df.columns]
            df_list.append(df_shifted)
    
    # If no DataFrames to concatenate, return empty DataFrame with same index
    if not df_list:
        return pd.DataFrame(index=df.index)
    
    # Concatenate all DataFrames
    return pd.concat(df_list, axis=1)


def generate_qTrans_Sub_Indices(X_transformed_train, y_train, h_qt=3, q_qt=0.25,h_sc=3, top_n=10000):
    """
    This function generates quantile transformed of individual series.
    Output from this function can be used for TFDI construction. Can be used as individual predictors
        - h_qt is the moving window size for quantile transformation.
        - q_qt is the quantile to use for transformation (default is 0.25), for pro-cyclical variables
        - h_sc is the moving window size for variable screening (default is 3)
        - top_n is the number of top variables to select per category (targeting; default is large so that we include all series)

    """
    
    # Generate Unweighted Sub-Indices
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
    # counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    counter_cyclical_vars = {'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'ISRATIOx', 'VIXCLSx'} #yield spreads are pro-cyclical (in that negative slope->recession)
    special_financial_vars = {'AAAFFM', 'BAAFFM', 'VIXCLSx'}
    X_momentum = X_transformed_train.rolling(window=h_sc, min_periods=1).mean()

    # Targeting (horizon=0 because we shifted x already)
    refined_variable_groups = select_top_variables_per_category(X_momentum, y_train, variable_groups, horizon=0, top_n=top_n, corr_threshold=0.0)
    
    all_selected_vars = [var for var_list in refined_variable_groups.values() for var in var_list]
    
    # Pre-allocate dictionaries to collect data
    weakness_data = {}
    deterioration_data = {}
    
    for var in all_selected_vars:
        signal_for_ranking = X_transformed_train[var]
        is_counter_theoretical = var in counter_cyclical_vars
        use_counter_logic = is_counter_theoretical

        # Use transformed values for special financial vars, not levels
        # We don't do this for a moment
        # if horizon == 1 and var in special_financial_vars:
        #     use_counter_logic = True
        #     signal_for_ranking = X_transformed_train[var].diff()
        # elif horizon > 1 and var in special_financial_vars:
        #     signal_for_ranking = X_transformed_train[var].diff()

        level_signal = signal_for_ranking
        momentum_signal = signal_for_ranking.rolling(window=h_qt, min_periods=1).mean()

        upper_quantile = 1.0 - q_qt
        lower_quantile = q_qt

        weakness_threshold = level_signal.quantile(upper_quantile if use_counter_logic else lower_quantile)
        weak_state = pd.Series(0.0, index=level_signal.index)
        if use_counter_logic: weak_state[level_signal > weakness_threshold] = 1.0
        else: weak_state[level_signal < weakness_threshold] = 1.0
        weakness_data[var] = weak_state

        deterioration_threshold = momentum_signal.quantile(upper_quantile if use_counter_logic else lower_quantile)
        deteriorating_state = pd.Series(0.0, index=momentum_signal.index)
        if use_counter_logic: deteriorating_state[momentum_signal > deterioration_threshold] = 1.0
        else: deteriorating_state[momentum_signal < deterioration_threshold] = 1.0
        deterioration_data[var] = deteriorating_state
    
    # Create DataFrames efficiently using pd.concat
    weakness_states = pd.DataFrame(weakness_data)
    deterioration_states = pd.DataFrame(deterioration_data)

    # Dataset for disaggregate deterioration status
    # Copy deterioration_states to all_disaggregates (matrix similar to X_transformed_train but with deterioration states)
    all_disaggregates = deterioration_states.copy()
    
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



    return all_disaggregates, all_sub_indices.fillna(method='ffill').fillna(0)
