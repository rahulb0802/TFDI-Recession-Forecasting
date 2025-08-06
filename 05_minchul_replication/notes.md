# Notes for model exploring
When H=3;
1. Best model is TFDI_dis:Logit_L1 (Brier: 0.036488) followed by TFDI_Full_pca:Logit_L1 (Brier: 0.045608), (see `main_recursive_forecasting.py`)
2. TFDI_pca performs better than PCA_Factors_8  (Brier: 0.049676 versus 0.053166)
3. TFDI_pca performs better than TFDI_avg (Brier: 0.049676 versus 0.063356)
4. When we are allowed to choose the regularization strength, TFDI_dis with L1 regularization is the best model and it leads to even lower Brier score (0.035091). But, our default choice is very close to this ex-post best case. TFDI_dis performs better than TFDI_dis_with_Full at their ex-post best regularization strength (0.035091 versus 0.045657),  (`main_regularization_check_L1.py`)
5. In general, L1 regularization performs better than L2 regularization (see `main_regularization_check_L2.py`). We really need to kick out some of irrelevant variables. For TFDI-dis, the expost best L2 version performs slightly better than TFDI_pca. 
6. TFDI_pca: we check h_qt and q_qt, h_qt=1 with q_qt=0.25 are the best expost. It is interesting to see that higher quantile value like 0.25 performs well. (`see main_quantile_check.py`) 
7. TFDI_pca: 2 works the best but not much variation across the number in that all beat TFDI_avg (`see main_nfactor_check.py`)
8. TFDI_avg: similar performance curvature found for h_qt and q_qt for TFDI_avg. h_qt=1 still leads to better performance, q_qt=0.3 was the best, though.

When varying h;
1. h=1: (best) TFDI_Full_pca  : Logit_L1; Then, TFDI_pca and TFDI_dis are roughly similar
2. h=3: Best model is TFDI_dis:Logit_L1 followed by TFDI_Full_pca:Logit_L1
3. h=6: TFDI_pca performed the best followed by TFDI_Full_pca 
4. h=12: TFDI_dis:L1 performs the best followed by TFDI_pca

---

1. TFDI-dis with L1 regularization leads to better performance (`main_regularization_check_L1.py`)
2. Better than L2 - Yes (`main_regularization_check_L2.py`)
3. Better than full covariates (L1, L2) - Yes
4. better than TFDI-dis with full covariates (L1, L2) - Yes
5. better than PCA of full covariates - Yes
6. average of TFDI-dis performs quite well similar to L2 but L1 version performs better
7. TFDI-dis-pca beats everything above (also, no regularization needed)
8. TFDI-dis-pca + Full-pca beats everything above
It seems like "at Risk" transformation contains valuable information

Now, the question remains:
1. at which quantile? (`see main_quantile_check.py`) -> interestingly, 0.25 performed the best 
2. number of factors (`see main_nfactor_check.py`) -> 8 or 9 factors are the best
3. "at Risk" transformation of raw PCA 

# Detailed info (h=3)
================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.049676 (n=420/420)
  Logit_L1       : 0.049387 (n=420/420)

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.063356 (n=420/420)
  Logit_L1       : 0.063719 (n=420/420)

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit          : 0.053166 (n=420/420)
  Logit_L1       : 0.053308 (n=420/420)

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.045939 (n=420/420)
  Logit_L1       : 0.045608 (n=420/420)

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit          : 0.062633 (n=420/420)
  Logit_L1       : 0.036488 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit          : 0.085899 (n=420/420)
  Logit_L1       : 0.051304 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit          : 0.166327 (n=420/420)
  Logit_L1       : 0.085698 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit          : 0.095954 (n=420/420)
  Logit_L1       : 0.096099 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit          : 0.062389 (n=420/420)
  Logit_L1       : 0.062895 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_pca    Logit         0.049676                420                420
          TFDI_pca Logit_L1         0.049387                420                420
          TFDI_avg    Logit         0.063356                420                420
          TFDI_avg Logit_L1         0.063719                420                420
     PCA_Factors_8    Logit         0.053166                420                420
     PCA_Factors_8 Logit_L1         0.053308                420                420
     TFDI_Full_pca    Logit         0.045939                420                420
     TFDI_Full_pca Logit_L1         0.045608                420                420
          TFDI_dis    Logit         0.062633                420                420
          TFDI_dis Logit_L1         0.036488                420                420
TFDI_dis_with_Full    Logit         0.085899                420                420
TFDI_dis_with_Full Logit_L1         0.051304                420                420
              Full    Logit         0.166327                420                420
              Full Logit_L1         0.085698                420                420
             Yield    Logit         0.095954                420                420
             Yield Logit_L1         0.096099                420                420
               ADS    Logit         0.062389                420                420
               ADS Logit_L1         0.062895                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_pca       : Logit_L1        (Brier: 0.049387)
TFDI_avg       : Logit           (Brier: 0.063356)
PCA_Factors_8  : Logit           (Brier: 0.053166)
TFDI_Full_pca  : Logit_L1        (Brier: 0.045608)
TFDI_dis       : Logit_L1        (Brier: 0.036488)
TFDI_dis_with_Full: Logit_L1        (Brier: 0.051304)
Full           : Logit_L1        (Brier: 0.085698)
Yield          : Logit           (Brier: 0.095954)
ADS            : Logit           (Brier: 0.062389)

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.175179 (n=420/420)
  Logit_L1_C0.022     : 0.107016 (n=420/420)
  Logit_L1_C0.060     : 0.061585 (n=420/420)
  Logit_L1_C0.167     : 0.039632 (n=420/420)
  Logit_L1_C0.464     : 0.035091 (n=420/420)
  Logit_L1_C1.292     : 0.037672 (n=420/420)
  Logit_L1_C3.594     : 0.044425 (n=420/420)
  Logit_L1_C10.000    : 0.053019 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.175151 (n=420/420)
  Logit_L1_C0.022     : 0.084443 (n=420/420)
  Logit_L1_C0.060     : 0.061142 (n=420/420)
  Logit_L1_C0.167     : 0.047843 (n=420/420)
  Logit_L1_C0.464     : 0.045657 (n=420/420)
  Logit_L1_C1.292     : 0.052977 (n=420/420)
  Logit_L1_C3.594     : 0.054232 (n=420/420)
  Logit_L1_C10.000    : 0.054229 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.175151 (n=420/420)
  Logit_L1_C0.022     : 0.084444 (n=420/420)
  Logit_L1_C0.060     : 0.061767 (n=420/420)
  Logit_L1_C0.167     : 0.059825 (n=420/420)
  Logit_L1_C0.464     : 0.072312 (n=420/420)
  Logit_L1_C1.292     : 0.088449 (n=420/420)
  Logit_L1_C3.594     : 0.097418 (n=420/420)
  Logit_L1_C10.000    : 0.115184 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set            Model   C_Value  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_dis  Logit_L1_C0.001  0.001000         0.250000                420                420
          TFDI_dis  Logit_L1_C0.003  0.003000         0.250000                420                420
          TFDI_dis  Logit_L1_C0.008  0.008000         0.175179                420                420
          TFDI_dis  Logit_L1_C0.022  0.022000         0.107016                420                420
          TFDI_dis  Logit_L1_C0.060  0.060000         0.061585                420                420
          TFDI_dis  Logit_L1_C0.167  0.167000         0.039632                420                420
          TFDI_dis  Logit_L1_C0.464  0.464000         0.035091                420                420
          TFDI_dis  Logit_L1_C1.292  1.292000         0.037672                420                420
          TFDI_dis  Logit_L1_C3.594  3.594000         0.044425                420                420
          TFDI_dis Logit_L1_C10.000 10.000000         0.053019                420                420
TFDI_dis_with_Full  Logit_L1_C0.001  0.001000         0.250000                420                420
TFDI_dis_with_Full  Logit_L1_C0.003  0.003000         0.250000                420                420
TFDI_dis_with_Full  Logit_L1_C0.008  0.008000         0.175151                420                420
TFDI_dis_with_Full  Logit_L1_C0.022  0.022000         0.084443                420                420
TFDI_dis_with_Full  Logit_L1_C0.060  0.060000         0.061142                420                420
TFDI_dis_with_Full  Logit_L1_C0.167  0.167000         0.047843                420                420
TFDI_dis_with_Full  Logit_L1_C0.464  0.464000         0.045657                420                420
TFDI_dis_with_Full  Logit_L1_C1.292  1.292000         0.052977                420                420
TFDI_dis_with_Full  Logit_L1_C3.594  3.594000         0.054232                420                420
TFDI_dis_with_Full Logit_L1_C10.000 10.000000         0.054229                420                420
              Full  Logit_L1_C0.001  0.001000         0.250000                420                420
              Full  Logit_L1_C0.003  0.003000         0.250000                420                420
              Full  Logit_L1_C0.008  0.008000         0.175151                420                420
              Full  Logit_L1_C0.022  0.022000         0.084444                420                420
              Full  Logit_L1_C0.060  0.060000         0.061767                420                420
              Full  Logit_L1_C0.167  0.167000         0.059825                420                420
              Full  Logit_L1_C0.464  0.464000         0.072312                420                420
              Full  Logit_L1_C1.292  1.292000         0.088449                420                420
              Full  Logit_L1_C3.594  3.594000         0.097418                420                420
              Full Logit_L1_C10.000 10.000000         0.115184                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis       : Logit_L1_C0.464      (C=0.464, Brier: 0.035091)
TFDI_dis_with_Full: Logit_L1_C0.464      (C=0.464, Brier: 0.045657)
Full           : Logit_L1_C0.167      (C=0.167, Brier: 0.059825)


================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit_L2_C0.001     : 0.162504 (n=420/420)
  Logit_L2_C0.003     : 0.124105 (n=420/420)
  Logit_L2_C0.008     : 0.090268 (n=420/420)
  Logit_L2_C0.022     : 0.066684 (n=420/420)
  Logit_L2_C0.060     : 0.053288 (n=420/420)
  Logit_L2_C0.167     : 0.046586 (n=420/420)
  Logit_L2_C0.464     : 0.044146 (n=420/420)
  Logit_L2_C1.292     : 0.045352 (n=420/420)
  Logit_L2_C3.594     : 0.049056 (n=420/420)
  Logit_L2_C10.000    : 0.054462 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit_L2_C0.001     : 0.129836 (n=420/420)
  Logit_L2_C0.003     : 0.091809 (n=420/420)
  Logit_L2_C0.008     : 0.071757 (n=420/420)
  Logit_L2_C0.022     : 0.063701 (n=420/420)
  Logit_L2_C0.060     : 0.061164 (n=420/420)
  Logit_L2_C0.167     : 0.060543 (n=420/420)
  Logit_L2_C0.464     : 0.061108 (n=420/420)
  Logit_L2_C1.292     : 0.063083 (n=420/420)
  Logit_L2_C3.594     : 0.066501 (n=420/420)
  Logit_L2_C10.000    : 0.070183 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit_L2_C0.001     : 0.194654 (n=420/420)
  Logit_L2_C0.003     : 0.152845 (n=420/420)
  Logit_L2_C0.008     : 0.110009 (n=420/420)
  Logit_L2_C0.022     : 0.083841 (n=420/420)
  Logit_L2_C0.060     : 0.076833 (n=420/420)
  Logit_L2_C0.167     : 0.079896 (n=420/420)
  Logit_L2_C0.464     : 0.086921 (n=420/420)
  Logit_L2_C1.292     : 0.095955 (n=420/420)
  Logit_L2_C3.594     : 0.106390 (n=420/420)
  Logit_L2_C10.000    : 0.118407 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set            Model   C_Value  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_dis  Logit_L2_C0.001  0.001000         0.162504                420                420
          TFDI_dis  Logit_L2_C0.003  0.003000         0.124105                420                420
          TFDI_dis  Logit_L2_C0.008  0.008000         0.090268                420                420
          TFDI_dis  Logit_L2_C0.022  0.022000         0.066684                420                420
          TFDI_dis  Logit_L2_C0.060  0.060000         0.053288                420                420
          TFDI_dis  Logit_L2_C0.167  0.167000         0.046586                420                420
          TFDI_dis  Logit_L2_C0.464  0.464000         0.044146                420                420
          TFDI_dis  Logit_L2_C1.292  1.292000         0.045352                420                420
          TFDI_dis  Logit_L2_C3.594  3.594000         0.049056                420                420
          TFDI_dis Logit_L2_C10.000 10.000000         0.054462                420                420
TFDI_dis_with_Full  Logit_L2_C0.001  0.001000         0.129836                420                420
TFDI_dis_with_Full  Logit_L2_C0.003  0.003000         0.091809                420                420
TFDI_dis_with_Full  Logit_L2_C0.008  0.008000         0.071757                420                420
TFDI_dis_with_Full  Logit_L2_C0.022  0.022000         0.063701                420                420
TFDI_dis_with_Full  Logit_L2_C0.060  0.060000         0.061164                420                420
TFDI_dis_with_Full  Logit_L2_C0.167  0.167000         0.060543                420                420
TFDI_dis_with_Full  Logit_L2_C0.464  0.464000         0.061108                420                420
TFDI_dis_with_Full  Logit_L2_C1.292  1.292000         0.063083                420                420
TFDI_dis_with_Full  Logit_L2_C3.594  3.594000         0.066501                420                420
TFDI_dis_with_Full Logit_L2_C10.000 10.000000         0.070183                420                420
              Full  Logit_L2_C0.001  0.001000         0.194654                420                420
              Full  Logit_L2_C0.003  0.003000         0.152845                420                420
              Full  Logit_L2_C0.008  0.008000         0.110009                420                420
              Full  Logit_L2_C0.022  0.022000         0.083841                420                420
              Full  Logit_L2_C0.060  0.060000         0.076833                420                420
              Full  Logit_L2_C0.167  0.167000         0.079896                420                420
              Full  Logit_L2_C0.464  0.464000         0.086921                420                420
              Full  Logit_L2_C1.292  1.292000         0.095955                420                420
              Full  Logit_L2_C3.594  3.594000         0.106390                420                420
              Full Logit_L2_C10.000 10.000000         0.118407                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis       : Logit_L2_C0.464      (C=0.464, Brier: 0.044146)
TFDI_dis_with_Full: Logit_L2_C0.167      (C=0.167, Brier: 0.060543)
Full           : Logit_L2_C0.060      (C=0.060, Brier: 0.076833)


================================================================================
ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  h_qt: 1
  q_qt: 0.25
  Predictor Set: TFDI_pca
  Brier Score: 0.047105

Summary by h_qt:
  h_qt=1: Average Brier Score = 0.056640
  h_qt=3: Average Brier Score = 0.057128
  h_qt=6: Average Brier Score = 0.058798

Summary by q_qt:
  q_qt=0.05: Average Brier Score = 0.085154
  q_qt=0.1: Average Brier Score = 0.070149
  q_qt=0.15: Average Brier Score = 0.057144
  q_qt=0.2: Average Brier Score = 0.050971
  q_qt=0.25: Average Brier Score = 0.049076
  q_qt=0.3: Average Brier Score = 0.048713
  q_qt=0.35: Average Brier Score = 0.048596
  q_qt=0.4: Average Brier Score = 0.050374

Summary by Predictor Set:
  TFDI_pca: Average Brier Score = 0.057522

Detailed Results:
 h_qt     q_qt Predictor_Set Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
    1 0.050000      TFDI_pca Logit         0.079713                420                420
    1 0.100000      TFDI_pca Logit         0.069204                420                420
    1 0.150000      TFDI_pca Logit         0.056229                420                420
    1 0.200000      TFDI_pca Logit         0.048266                420                420
    1 0.250000      TFDI_pca Logit         0.047105                420                420
    1 0.300000      TFDI_pca Logit         0.049063                420                420
    1 0.350000      TFDI_pca Logit         0.049983                420                420
    1 0.400000      TFDI_pca Logit         0.053560                420                420
    3 0.050000      TFDI_pca Logit         0.088663                420                420
    3 0.100000      TFDI_pca Logit         0.066963                420                420
    3 0.150000      TFDI_pca Logit         0.055726                420                420
    3 0.200000      TFDI_pca Logit         0.049468                420                420
    3 0.250000      TFDI_pca Logit         0.049774                420                420
    3 0.300000      TFDI_pca Logit         0.048893                420                420
    3 0.350000      TFDI_pca Logit         0.048593                420                420
    3 0.400000      TFDI_pca Logit         0.048943                420                420
    6 0.050000      TFDI_pca Logit         0.087087                420                420
    6 0.100000      TFDI_pca Logit         0.074280                420                420
    6 0.150000      TFDI_pca Logit         0.059477                420                420
    6 0.200000      TFDI_pca Logit         0.055178                420                420
    6 0.250000      TFDI_pca Logit         0.050349                420                420
    6 0.300000      TFDI_pca Logit         0.048183                420                420
    6 0.350000      TFDI_pca Logit         0.047212                420                420
    6 0.400000      TFDI_pca Logit         0.048618                420                420

================================================================================
ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  h_qt: 1
  q_qt: 0.3
  Predictor Set: TFDI_avg
  Brier Score: 0.062550

Summary by h_qt:
  h_qt=1: Average Brier Score = 0.068735
  h_qt=3: Average Brier Score = 0.068033
  h_qt=6: Average Brier Score = 0.072613

Summary by q_qt:
  q_qt=0.05: Average Brier Score = 0.083894
  q_qt=0.1: Average Brier Score = 0.075639
  q_qt=0.15: Average Brier Score = 0.070004
  q_qt=0.2: Average Brier Score = 0.066614
  q_qt=0.25: Average Brier Score = 0.065650
  q_qt=0.3: Average Brier Score = 0.064689
  q_qt=0.35: Average Brier Score = 0.065043
  q_qt=0.4: Average Brier Score = 0.066817

Summary by Predictor Set:
  TFDI_avg: Average Brier Score = 0.069794

Detailed Results:
 h_qt     q_qt Predictor_Set Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
    1 0.050000      TFDI_avg Logit         0.084086                420                420
    1 0.100000      TFDI_avg Logit         0.074596                420                420
    1 0.150000      TFDI_avg Logit         0.068917                420                420
    1 0.200000      TFDI_avg Logit         0.065326                420                420
    1 0.250000      TFDI_avg Logit         0.064711                420                420
    1 0.300000      TFDI_avg Logit         0.062550                420                420
    1 0.350000      TFDI_avg Logit         0.063348                420                420
    1 0.400000      TFDI_avg Logit         0.066349                420                420
    3 0.050000      TFDI_avg Logit         0.081891                420                420
    3 0.100000      TFDI_avg Logit         0.073150                420                420
    3 0.150000      TFDI_avg Logit         0.067217                420                420
    3 0.200000      TFDI_avg Logit         0.065004                420                420
    3 0.250000      TFDI_avg Logit         0.063356                420                420
    3 0.300000      TFDI_avg Logit         0.063203                420                420
    3 0.350000      TFDI_avg Logit         0.064686                420                420
    3 0.400000      TFDI_avg Logit         0.065760                420                420
    6 0.050000      TFDI_avg Logit         0.085704                420                420
    6 0.100000      TFDI_avg Logit         0.079172                420                420
    6 0.150000      TFDI_avg Logit         0.073879                420                420
    6 0.200000      TFDI_avg Logit         0.069512                420                420
    6 0.250000      TFDI_avg Logit         0.068883                420                420
    6 0.300000      TFDI_avg Logit         0.068316                420                420
    6 0.350000      TFDI_avg Logit         0.067095                420                420
    6 0.400000      TFDI_avg Logit         0.068343                420                420

================================================================================
FACTOR ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  n_factors: 2
  Brier Score: 0.047254

Summary statistics:
  Mean Brier Score: 0.050961
  Std Brier Score: 0.003261
  Min Brier Score: 0.047254
  Max Brier Score: 0.060234

Performance by number of factors:
  n_factors= 1: Brier Score = 0.060234
  n_factors= 2: Brier Score = 0.047254
  n_factors= 3: Brier Score = 0.049173
  n_factors= 4: Brier Score = 0.051764
  n_factors= 5: Brier Score = 0.053029
  n_factors= 6: Brier Score = 0.049007
  n_factors= 7: Brier Score = 0.050146
  n_factors= 8: Brier Score = 0.049751
  n_factors= 9: Brier Score = 0.050401
  n_factors=10: Brier Score = 0.050123
  n_factors=11: Brier Score = 0.049493
  n_factors=12: Brier Score = 0.051155

Optimal number of factors: 2

Detailed Results:
 n_factors  Avg_Brier_Score  Valid_Predictions  Total_Predictions
         1         0.060234                420                420
         2         0.047254                420                420
         3         0.049173                420                420
         4         0.051764                420                420
         5         0.053029                420                420
         6         0.049007                420                420
         7         0.050146                420                420
         8         0.049751                420                420
         9         0.050401                420                420
        10         0.050123                420                420
        11         0.049493                420                420
        12         0.051155                420                420

---
H=6

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.067333 (n=420/420)
  Logit_L1       : 0.066837 (n=420/420)

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.079168 (n=420/420)
  Logit_L1       : 0.079424 (n=420/420)

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit          : 0.070265 (n=420/420)
  Logit_L1       : 0.070260 (n=420/420)

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.070221 (n=420/420)
  Logit_L1       : 0.069663 (n=420/420)

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit          : 0.144389 (n=420/420)
  Logit_L1       : 0.073113 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit          : 0.125246 (n=420/420)
  Logit_L1       : 0.077648 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit          : 0.150403 (n=420/420)
  Logit_L1       : 0.086428 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit          : 0.093240 (n=420/420)
  Logit_L1       : 0.093302 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit          : 0.082972 (n=420/420)
  Logit_L1       : 0.083281 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_pca    Logit         0.067333                420                420
          TFDI_pca Logit_L1         0.066837                420                420
          TFDI_avg    Logit         0.079168                420                420
          TFDI_avg Logit_L1         0.079424                420                420
     PCA_Factors_8    Logit         0.070265                420                420
     PCA_Factors_8 Logit_L1         0.070260                420                420
     TFDI_Full_pca    Logit         0.070221                420                420
     TFDI_Full_pca Logit_L1         0.069663                420                420
          TFDI_dis    Logit         0.144389                420                420
          TFDI_dis Logit_L1         0.073113                420                420
TFDI_dis_with_Full    Logit         0.125246                420                420
TFDI_dis_with_Full Logit_L1         0.077648                420                420
              Full    Logit         0.150403                420                420
              Full Logit_L1         0.086428                420                420
             Yield    Logit         0.093240                420                420
             Yield Logit_L1         0.093302                420                420
               ADS    Logit         0.082972                420                420
               ADS Logit_L1         0.083281                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_pca       : Logit_L1        (Brier: 0.066837)
TFDI_avg       : Logit           (Brier: 0.079168)
PCA_Factors_8  : Logit_L1        (Brier: 0.070260)
TFDI_Full_pca  : Logit_L1        (Brier: 0.069663)
TFDI_dis       : Logit_L1        (Brier: 0.073113)
TFDI_dis_with_Full: Logit_L1        (Brier: 0.077648)
Full           : Logit_L1        (Brier: 0.086428)
Yield          : Logit           (Brier: 0.093240)
ADS            : Logit           (Brier: 0.082972)

---
H=12
================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.073299 (n=420/420)
  Logit_L1       : 0.073103 (n=420/420)

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.089287 (n=420/420)
  Logit_L1       : 0.089735 (n=420/420)

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit          : 0.074128 (n=420/420)
  Logit_L1       : 0.074082 (n=420/420)

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.077620 (n=420/420)
  Logit_L1       : 0.076493 (n=420/420)

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit          : 0.110160 (n=420/420)
  Logit_L1       : 0.056643 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit          : 0.147550 (n=420/420)
  Logit_L1       : 0.079231 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit          : 0.164678 (n=420/420)
  Logit_L1       : 0.109227 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit          : 0.080636 (n=420/420)
  Logit_L1       : 0.080534 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit          : 0.089940 (n=420/420)
  Logit_L1       : 0.089781 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_pca    Logit         0.073299                420                420
          TFDI_pca Logit_L1         0.073103                420                420
          TFDI_avg    Logit         0.089287                420                420
          TFDI_avg Logit_L1         0.089735                420                420
     PCA_Factors_8    Logit         0.074128                420                420
     PCA_Factors_8 Logit_L1         0.074082                420                420
     TFDI_Full_pca    Logit         0.077620                420                420
     TFDI_Full_pca Logit_L1         0.076493                420                420
          TFDI_dis    Logit         0.110160                420                420
          TFDI_dis Logit_L1         0.056643                420                420
TFDI_dis_with_Full    Logit         0.147550                420                420
TFDI_dis_with_Full Logit_L1         0.079231                420                420
              Full    Logit         0.164678                420                420
              Full Logit_L1         0.109227                420                420
             Yield    Logit         0.080636                420                420
             Yield Logit_L1         0.080534                420                420
               ADS    Logit         0.089940                420                420
               ADS Logit_L1         0.089781                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_pca       : Logit_L1        (Brier: 0.073103)
TFDI_avg       : Logit           (Brier: 0.089287)
PCA_Factors_8  : Logit_L1        (Brier: 0.074082)
TFDI_Full_pca  : Logit_L1        (Brier: 0.076493)
TFDI_dis       : Logit_L1        (Brier: 0.056643)
TFDI_dis_with_Full: Logit_L1        (Brier: 0.079231)
Full           : Logit_L1        (Brier: 0.109227)
Yield          : Logit_L1        (Brier: 0.080534)
ADS            : Logit_L1        (Brier: 0.089781)

---
H=1

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.030119 (n=420/420)
  Logit_L1       : 0.029631 (n=420/420)

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.049052 (n=420/420)
  Logit_L1       : 0.049086 (n=420/420)

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit          : 0.034321 (n=420/420)
  Logit_L1       : 0.034950 (n=420/420)

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.027476 (n=420/420)
  Logit_L1       : 0.027086 (n=420/420)

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit          : 0.049832 (n=420/420)
  Logit_L1       : 0.029706 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit          : 0.061678 (n=420/420)
  Logit_L1       : 0.039971 (n=420/420)

Predictor Set: Full
--------------------------------------------------
  Logit          : 0.093156 (n=420/420)
  Logit_L1       : 0.058409 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit          : 0.092530 (n=420/420)
  Logit_L1       : 0.092707 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit          : 0.037400 (n=420/420)
  Logit_L1       : 0.038168 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_pca    Logit         0.030119                420                420
          TFDI_pca Logit_L1         0.029631                420                420
          TFDI_avg    Logit         0.049052                420                420
          TFDI_avg Logit_L1         0.049086                420                420
     PCA_Factors_8    Logit         0.034321                420                420
     PCA_Factors_8 Logit_L1         0.034950                420                420
     TFDI_Full_pca    Logit         0.027476                420                420
     TFDI_Full_pca Logit_L1         0.027086                420                420
          TFDI_dis    Logit         0.049832                420                420
          TFDI_dis Logit_L1         0.029706                420                420
TFDI_dis_with_Full    Logit         0.061678                420                420
TFDI_dis_with_Full Logit_L1         0.039971                420                420
              Full    Logit         0.093156                420                420
              Full Logit_L1         0.058409                420                420
             Yield    Logit         0.092530                420                420
             Yield Logit_L1         0.092707                420                420
               ADS    Logit         0.037400                420                420
               ADS Logit_L1         0.038168                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_pca       : Logit_L1        (Brier: 0.029631)
TFDI_avg       : Logit           (Brier: 0.049052)
PCA_Factors_8  : Logit           (Brier: 0.034321)
TFDI_Full_pca  : Logit_L1        (Brier: 0.027086)
TFDI_dis       : Logit_L1        (Brier: 0.029706)
TFDI_dis_with_Full: Logit_L1        (Brier: 0.039971)
Full           : Logit_L1        (Brier: 0.058409)
Yield          : Logit           (Brier: 0.092530)
ADS            : Logit           (Brier: 0.037400)