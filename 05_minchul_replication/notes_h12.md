# Notes for h=12

- TFDI_dis performs best
- Then, TFDI_pca performs well
  -> longer horizon, selection may be useful (L1 reguarlization);
  -> 

# details
================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.077494 (n=420/420)
  Logit_L1       : 0.076392 (n=420/420)

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.073240 (n=420/420)
  Logit_L1       : 0.073047 (n=420/420)

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.089287 (n=420/420)
  Logit_L1       : 0.089735 (n=420/420)

Predictor Set: TFDI_dist
--------------------------------------------------
  Logit          : No valid predictions
  Logit_L1       : No valid predictions

Predictor Set: TFDI_dist_with_Full
--------------------------------------------------
  Logit          : No valid predictions
  Logit_L1       : No valid predictions

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit          : 0.074101 (n=420/420)
  Logit_L1       : 0.074055 (n=420/420)

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
      TFDI_Full_pca    Logit         0.077494                420                420
      TFDI_Full_pca Logit_L1         0.076392                420                420
           TFDI_pca    Logit         0.073240                420                420
           TFDI_pca Logit_L1         0.073047                420                420
           TFDI_avg    Logit         0.089287                420                420
           TFDI_avg Logit_L1         0.089735                420                420
          TFDI_dist    Logit              NaN                  0                420
          TFDI_dist Logit_L1              NaN                  0                420
TFDI_dist_with_Full    Logit              NaN                  0                420
TFDI_dist_with_Full Logit_L1              NaN                  0                420
      PCA_Factors_8    Logit         0.074101                420                420
      PCA_Factors_8 Logit_L1         0.074055                420                420
              Yield    Logit         0.080636                420                420
              Yield Logit_L1         0.080534                420                420
                ADS    Logit         0.089940                420                420
                ADS Logit_L1         0.089781                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_Full_pca  : Logit_L1        (Brier: 0.076392)
TFDI_pca       : Logit_L1        (Brier: 0.073047)
TFDI_avg       : Logit           (Brier: 0.089287)
TFDI_dist      : No valid predictions
TFDI_dist_with_Full: No valid predictions
PCA_Factors_8  : Logit_L1        (Brier: 0.074055)
Yield          : Logit_L1        (Brier: 0.080534)
ADS            : Logit_L1        (Brier: 0.089781)

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.175179 (n=420/420)
  Logit_L1_C0.022     : 0.107017 (n=420/420)
  Logit_L1_C0.060     : 0.075971 (n=420/420)
  Logit_L1_C0.167     : 0.063026 (n=420/420)
  Logit_L1_C0.464     : 0.056280 (n=420/420)
  Logit_L1_C1.292     : 0.058482 (n=420/420)
  Logit_L1_C3.594     : 0.070224 (n=420/420)
  Logit_L1_C10.000    : 0.086403 (n=420/420)

Predictor Set: TFDI_dis_with_Full
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.175179 (n=420/420)
  Logit_L1_C0.022     : 0.091265 (n=420/420)
  Logit_L1_C0.060     : 0.072530 (n=420/420)
  Logit_L1_C0.167     : 0.069098 (n=420/420)
  Logit_L1_C0.464     : 0.068708 (n=420/420)
  Logit_L1_C1.292     : 0.084298 (n=420/420)
  Logit_L1_C3.594     : 0.099174 (n=420/420)
  Logit_L1_C10.000    : 0.106895 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set            Model   C_Value  Avg_Brier_Score  Valid_Predictions  Total_Predictions
          TFDI_dis  Logit_L1_C0.001  0.001000         0.250000                420                420
          TFDI_dis  Logit_L1_C0.003  0.003000         0.250000                420                420
          TFDI_dis  Logit_L1_C0.008  0.008000         0.175179                420                420
          TFDI_dis  Logit_L1_C0.022  0.022000         0.107017                420                420
          TFDI_dis  Logit_L1_C0.060  0.060000         0.075971                420                420
          TFDI_dis  Logit_L1_C0.167  0.167000         0.063026                420                420
          TFDI_dis  Logit_L1_C0.464  0.464000         0.056280                420                420
          TFDI_dis  Logit_L1_C1.292  1.292000         0.058482                420                420
          TFDI_dis  Logit_L1_C3.594  3.594000         0.070224                420                420
          TFDI_dis Logit_L1_C10.000 10.000000         0.086403                420                420
TFDI_dis_with_Full  Logit_L1_C0.001  0.001000         0.250000                420                420
TFDI_dis_with_Full  Logit_L1_C0.003  0.003000         0.250000                420                420
TFDI_dis_with_Full  Logit_L1_C0.008  0.008000         0.175179                420                420
TFDI_dis_with_Full  Logit_L1_C0.022  0.022000         0.091265                420                420
TFDI_dis_with_Full  Logit_L1_C0.060  0.060000         0.072530                420                420
TFDI_dis_with_Full  Logit_L1_C0.167  0.167000         0.069098                420                420
TFDI_dis_with_Full  Logit_L1_C0.464  0.464000         0.068708                420                420
TFDI_dis_with_Full  Logit_L1_C1.292  1.292000         0.084298                420                420
TFDI_dis_with_Full  Logit_L1_C3.594  3.594000         0.099174                420                420
TFDI_dis_with_Full Logit_L1_C10.000 10.000000         0.106895                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis       : Logit_L1_C0.464      (C=0.464, Brier: 0.056280)
TFDI_dis_with_Full: Logit_L1_C0.464      (C=0.464, Brier: 0.068708)
