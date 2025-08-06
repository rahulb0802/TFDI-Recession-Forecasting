# Notes for model exploring
When H=3;
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
1] at which quantile? (`see main_quantile_check.py`) -> interestingly, 0.25 performed the best 
2] number of factors (`see main_nfactor_check.py`) -> 8 or 9 factors are the best
3] "at Risk" transformation of raw PCA 

# Detailed info

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

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

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.251999 (n=420/420)
  Logit_L1_C0.008     : 0.162099 (n=420/420)
  Logit_L1_C0.022     : 0.078410 (n=420/420)
  Logit_L1_C0.060     : 0.058974 (n=420/420)
  Logit_L1_C0.167     : 0.054431 (n=420/420)
  Logit_L1_C0.464     : 0.053535 (n=420/420)
  Logit_L1_C1.292     : 0.053297 (n=420/420)
  Logit_L1_C3.594     : 0.053228 (n=420/420)
  Logit_L1_C10.000    : 0.053205 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.243025 (n=420/420)
  Logit_L1_C0.008     : 0.158604 (n=420/420)
  Logit_L1_C0.022     : 0.110197 (n=420/420)
  Logit_L1_C0.060     : 0.099284 (n=420/420)
  Logit_L1_C0.167     : 0.096918 (n=420/420)
  Logit_L1_C0.464     : 0.096272 (n=420/420)
  Logit_L1_C1.292     : 0.096066 (n=420/420)
  Logit_L1_C3.594     : 0.095995 (n=420/420)
  Logit_L1_C10.000    : 0.095971 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit_L1_C0.001     : 0.250000 (n=420/420)
  Logit_L1_C0.003     : 0.250000 (n=420/420)
  Logit_L1_C0.008     : 0.176323 (n=420/420)
  Logit_L1_C0.022     : 0.096537 (n=420/420)
  Logit_L1_C0.060     : 0.072458 (n=420/420)
  Logit_L1_C0.167     : 0.065628 (n=420/420)
  Logit_L1_C0.464     : 0.063495 (n=420/420)
  Logit_L1_C1.292     : 0.062780 (n=420/420)
  Logit_L1_C3.594     : 0.062531 (n=420/420)
  Logit_L1_C10.000    : 0.062442 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set            Model   C_Value  Avg_Brier_Score  Valid_Predictions  Total_Predictions
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
     PCA_Factors_8  Logit_L1_C0.001  0.001000         0.250000                420                420
     PCA_Factors_8  Logit_L1_C0.003  0.003000         0.251999                420                420
     PCA_Factors_8  Logit_L1_C0.008  0.008000         0.162099                420                420
     PCA_Factors_8  Logit_L1_C0.022  0.022000         0.078410                420                420
     PCA_Factors_8  Logit_L1_C0.060  0.060000         0.058974                420                420
     PCA_Factors_8  Logit_L1_C0.167  0.167000         0.054431                420                420
     PCA_Factors_8  Logit_L1_C0.464  0.464000         0.053535                420                420
     PCA_Factors_8  Logit_L1_C1.292  1.292000         0.053297                420                420
     PCA_Factors_8  Logit_L1_C3.594  3.594000         0.053228                420                420
     PCA_Factors_8 Logit_L1_C10.000 10.000000         0.053205                420                420
             Yield  Logit_L1_C0.001  0.001000         0.250000                420                420
             Yield  Logit_L1_C0.003  0.003000         0.243025                420                420
             Yield  Logit_L1_C0.008  0.008000         0.158604                420                420
             Yield  Logit_L1_C0.022  0.022000         0.110197                420                420
             Yield  Logit_L1_C0.060  0.060000         0.099284                420                420
             Yield  Logit_L1_C0.167  0.167000         0.096918                420                420
             Yield  Logit_L1_C0.464  0.464000         0.096272                420                420
             Yield  Logit_L1_C1.292  1.292000         0.096066                420                420
             Yield  Logit_L1_C3.594  3.594000         0.095995                420                420
             Yield Logit_L1_C10.000 10.000000         0.095971                420                420
               ADS  Logit_L1_C0.001  0.001000         0.250000                420                420
               ADS  Logit_L1_C0.003  0.003000         0.250000                420                420
               ADS  Logit_L1_C0.008  0.008000         0.176323                420                420
               ADS  Logit_L1_C0.022  0.022000         0.096537                420                420
               ADS  Logit_L1_C0.060  0.060000         0.072458                420                420
               ADS  Logit_L1_C0.167  0.167000         0.065628                420                420
               ADS  Logit_L1_C0.464  0.464000         0.063495                420                420
               ADS  Logit_L1_C1.292  1.292000         0.062780                420                420
               ADS  Logit_L1_C3.594  3.594000         0.062531                420                420
               ADS Logit_L1_C10.000 10.000000         0.062442                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis_with_Full: Logit_L1_C0.464      (C=0.464, Brier: 0.045657)
Full           : Logit_L1_C0.167      (C=0.167, Brier: 0.059825)
PCA_Factors_8  : Logit_L1_C10.000     (C=10.000, Brier: 0.053205)
Yield          : Logit_L1_C10.000     (C=10.000, Brier: 0.095971)
ADS            : Logit_L1_C10.000     (C=10.000, Brier: 0.062442)


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
  Logit_L2_C1.292     : 0.045353 (n=420/420)
  Logit_L2_C3.594     : 0.049056 (n=420/420)
  Logit_L2_C10.000    : 0.054460 (n=420/420)

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
     TFDI_dis  Logit_L2_C1.292  1.292000         0.045353                420                420
     TFDI_dis  Logit_L2_C3.594  3.594000         0.049056                420                420
     TFDI_dis Logit_L2_C10.000 10.000000         0.054460                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis       : Logit_L2_C0.464      (C=0.464, Brier: 0.044146)


================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

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
  Logit_L2_C3.594     : 0.066505 (n=420/420)
  Logit_L2_C10.000    : 0.070183 (n=420/420)

Predictor Set: TFDI_dis
--------------------------------------------------
  Logit_L2_C0.001     : 0.162504 (n=420/420)
  Logit_L2_C0.003     : 0.124105 (n=420/420)
  Logit_L2_C0.008     : 0.090268 (n=420/420)
  Logit_L2_C0.022     : 0.066684 (n=420/420)
  Logit_L2_C0.060     : 0.053288 (n=420/420)
  Logit_L2_C0.167     : 0.046586 (n=420/420)
  Logit_L2_C0.464     : 0.044146 (n=420/420)
  Logit_L2_C1.292     : 0.045353 (n=420/420)
  Logit_L2_C3.594     : 0.049056 (n=420/420)
  Logit_L2_C10.000    : 0.054460 (n=420/420)

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
  Logit_L2_C3.594     : 0.106391 (n=420/420)
  Logit_L2_C10.000    : 0.118404 (n=420/420)

Predictor Set: PCA_Factors_8
--------------------------------------------------
  Logit_L2_C0.001     : 0.196138 (n=420/420)
  Logit_L2_C0.003     : 0.155405 (n=420/420)
  Logit_L2_C0.008     : 0.110896 (n=420/420)
  Logit_L2_C0.022     : 0.078611 (n=420/420)
  Logit_L2_C0.060     : 0.062610 (n=420/420)
  Logit_L2_C0.167     : 0.056384 (n=420/420)
  Logit_L2_C0.464     : 0.054236 (n=420/420)
  Logit_L2_C1.292     : 0.053530 (n=420/420)
  Logit_L2_C3.594     : 0.053295 (n=420/420)
  Logit_L2_C10.000    : 0.053215 (n=420/420)

Predictor Set: Yield
--------------------------------------------------
  Logit_L2_C0.001     : 0.176717 (n=420/420)
  Logit_L2_C0.003     : 0.141319 (n=420/420)
  Logit_L2_C0.008     : 0.118488 (n=420/420)
  Logit_L2_C0.022     : 0.106314 (n=420/420)
  Logit_L2_C0.060     : 0.100238 (n=420/420)
  Logit_L2_C0.167     : 0.097585 (n=420/420)
  Logit_L2_C0.464     : 0.096553 (n=420/420)
  Logit_L2_C1.292     : 0.096172 (n=420/420)
  Logit_L2_C3.594     : 0.096034 (n=420/420)
  Logit_L2_C10.000    : 0.095984 (n=420/420)

Predictor Set: ADS
--------------------------------------------------
  Logit_L2_C0.001     : 0.216236 (n=420/420)
  Logit_L2_C0.003     : 0.178757 (n=420/420)
  Logit_L2_C0.008     : 0.131793 (n=420/420)
  Logit_L2_C0.022     : 0.095717 (n=420/420)
  Logit_L2_C0.060     : 0.076410 (n=420/420)
  Logit_L2_C0.167     : 0.067880 (n=420/420)
  Logit_L2_C0.464     : 0.064450 (n=420/420)
  Logit_L2_C1.292     : 0.063146 (n=420/420)
  Logit_L2_C3.594     : 0.062665 (n=420/420)
  Logit_L2_C10.000    : 0.062490 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
     Predictor_Set            Model   C_Value  Avg_Brier_Score  Valid_Predictions  Total_Predictions
TFDI_dis_with_Full  Logit_L2_C0.001  0.001000         0.129836                420                420
TFDI_dis_with_Full  Logit_L2_C0.003  0.003000         0.091809                420                420
TFDI_dis_with_Full  Logit_L2_C0.008  0.008000         0.071757                420                420
TFDI_dis_with_Full  Logit_L2_C0.022  0.022000         0.063701                420                420
TFDI_dis_with_Full  Logit_L2_C0.060  0.060000         0.061164                420                420
TFDI_dis_with_Full  Logit_L2_C0.167  0.167000         0.060543                420                420
TFDI_dis_with_Full  Logit_L2_C0.464  0.464000         0.061108                420                420
TFDI_dis_with_Full  Logit_L2_C1.292  1.292000         0.063083                420                420
TFDI_dis_with_Full  Logit_L2_C3.594  3.594000         0.066505                420                420
TFDI_dis_with_Full Logit_L2_C10.000 10.000000         0.070183                420                420
          TFDI_dis  Logit_L2_C0.001  0.001000         0.162504                420                420
          TFDI_dis  Logit_L2_C0.003  0.003000         0.124105                420                420
          TFDI_dis  Logit_L2_C0.008  0.008000         0.090268                420                420
          TFDI_dis  Logit_L2_C0.022  0.022000         0.066684                420                420
          TFDI_dis  Logit_L2_C0.060  0.060000         0.053288                420                420
          TFDI_dis  Logit_L2_C0.167  0.167000         0.046586                420                420
          TFDI_dis  Logit_L2_C0.464  0.464000         0.044146                420                420
          TFDI_dis  Logit_L2_C1.292  1.292000         0.045353                420                420
          TFDI_dis  Logit_L2_C3.594  3.594000         0.049056                420                420
          TFDI_dis Logit_L2_C10.000 10.000000         0.054460                420                420
              Full  Logit_L2_C0.001  0.001000         0.194654                420                420
              Full  Logit_L2_C0.003  0.003000         0.152845                420                420
              Full  Logit_L2_C0.008  0.008000         0.110009                420                420
              Full  Logit_L2_C0.022  0.022000         0.083841                420                420
              Full  Logit_L2_C0.060  0.060000         0.076833                420                420
              Full  Logit_L2_C0.167  0.167000         0.079896                420                420
              Full  Logit_L2_C0.464  0.464000         0.086921                420                420
              Full  Logit_L2_C1.292  1.292000         0.095955                420                420
              Full  Logit_L2_C3.594  3.594000         0.106391                420                420
              Full Logit_L2_C10.000 10.000000         0.118404                420                420
     PCA_Factors_8  Logit_L2_C0.001  0.001000         0.196138                420                420
     PCA_Factors_8  Logit_L2_C0.003  0.003000         0.155405                420                420
     PCA_Factors_8  Logit_L2_C0.008  0.008000         0.110896                420                420
     PCA_Factors_8  Logit_L2_C0.022  0.022000         0.078611                420                420
     PCA_Factors_8  Logit_L2_C0.060  0.060000         0.062610                420                420
     PCA_Factors_8  Logit_L2_C0.167  0.167000         0.056384                420                420
     PCA_Factors_8  Logit_L2_C0.464  0.464000         0.054236                420                420
     PCA_Factors_8  Logit_L2_C1.292  1.292000         0.053530                420                420
     PCA_Factors_8  Logit_L2_C3.594  3.594000         0.053295                420                420
     PCA_Factors_8 Logit_L2_C10.000 10.000000         0.053215                420                420
             Yield  Logit_L2_C0.001  0.001000         0.176717                420                420
             Yield  Logit_L2_C0.003  0.003000         0.141319                420                420
             Yield  Logit_L2_C0.008  0.008000         0.118488                420                420
             Yield  Logit_L2_C0.022  0.022000         0.106314                420                420
             Yield  Logit_L2_C0.060  0.060000         0.100238                420                420
             Yield  Logit_L2_C0.167  0.167000         0.097585                420                420
             Yield  Logit_L2_C0.464  0.464000         0.096553                420                420
             Yield  Logit_L2_C1.292  1.292000         0.096172                420                420
             Yield  Logit_L2_C3.594  3.594000         0.096034                420                420
             Yield Logit_L2_C10.000 10.000000         0.095984                420                420
               ADS  Logit_L2_C0.001  0.001000         0.216236                420                420
               ADS  Logit_L2_C0.003  0.003000         0.178757                420                420
               ADS  Logit_L2_C0.008  0.008000         0.131793                420                420
               ADS  Logit_L2_C0.022  0.022000         0.095717                420                420
               ADS  Logit_L2_C0.060  0.060000         0.076410                420                420
               ADS  Logit_L2_C0.167  0.167000         0.067880                420                420
               ADS  Logit_L2_C0.464  0.464000         0.064450                420                420
               ADS  Logit_L2_C1.292  1.292000         0.063146                420                420
               ADS  Logit_L2_C3.594  3.594000         0.062665                420                420
               ADS Logit_L2_C10.000 10.000000         0.062490                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_dis_with_Full: Logit_L2_C0.167      (C=0.167, Brier: 0.060543)
TFDI_dis       : Logit_L2_C0.464      (C=0.464, Brier: 0.044146)
Full           : Logit_L2_C0.060      (C=0.060, Brier: 0.076833)
PCA_Factors_8  : Logit_L2_C10.000     (C=10.000, Brier: 0.053215)
Yield          : Logit_L2_C10.000     (C=10.000, Brier: 0.095984)
ADS            : Logit_L2_C10.000     (C=10.000, Brier: 0.062490)

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.049052 (n=420/420)
  Logit_L1       : 0.049086 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
     TFDI_avg    Logit         0.049052                420                420
     TFDI_avg Logit_L1         0.049086                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_avg       : Logit           (Brier: 0.049052)


================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_avg
--------------------------------------------------
  Logit          : 0.049052 (n=420/420)
  Logit_L1       : 0.049086 (n=420/420)

Predictor Set: TFDI_pca
--------------------------------------------------
  Logit          : 0.030144 (n=420/420)
  Logit_L1       : 0.029663 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
     TFDI_avg    Logit         0.049052                420                420
     TFDI_avg Logit_L1         0.049086                420                420
     TFDI_pca    Logit         0.030144                420                420
     TFDI_pca Logit_L1         0.029663                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_avg       : Logit           (Brier: 0.049052)
TFDI_pca       : Logit_L1        (Brier: 0.029663)

================================================================================
AVERAGE BRIER SCORES (OUT-OF-SAMPLE ERRORS)
================================================================================

Predictor Set: TFDI_Full_pca
--------------------------------------------------
  Logit          : 0.027572 (n=420/420)
  Logit_L1       : 0.027200 (n=420/420)

================================================================================
SUMMARY TABLE
================================================================================
Predictor_Set    Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
TFDI_Full_pca    Logit         0.027572                420                420
TFDI_Full_pca Logit_L1         0.027200                420                420

================================================================================
BEST PERFORMING MODELS BY PREDICTOR SET
================================================================================
TFDI_Full_pca  : Logit_L1        (Brier: 0.027200)


# Robustness

================================================================================
ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  h_qt: 3
  q_qt: 0.25
  Predictor Set: TFDI_Full_pca
  Brier Score: 0.027690

Summary by h_qt:
  h_qt=3: Average Brier Score = 0.045435
  h_qt=12: Average Brier Score = 0.061715

Summary by q_qt:
  q_qt=0.05: Average Brier Score = 0.063972
  q_qt=0.1: Average Brier Score = 0.052554
  q_qt=0.25: Average Brier Score = 0.044199

Summary by Predictor Set:
  TFDI_Full_pca: Average Brier Score = 0.036754
  TFDI_pca: Average Brier Score = 0.056367
  TFDI_avg: Average Brier Score = 0.067604

Detailed Results:
 h_qt     q_qt Predictor_Set Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
    3 0.050000 TFDI_Full_pca Logit         0.040166                420                420
    3 0.050000      TFDI_pca Logit         0.057934                420                420
    3 0.050000      TFDI_avg Logit         0.067682                420                420
    3 0.100000 TFDI_Full_pca Logit         0.037693                420                420
    3 0.100000      TFDI_pca Logit         0.042607                420                420
    3 0.100000      TFDI_avg Logit         0.055956                420                420
    3 0.250000 TFDI_Full_pca Logit         0.027690                420                420
    3 0.250000      TFDI_pca Logit         0.030130                420                420
    3 0.250000      TFDI_avg Logit         0.049052                420                420
   12 0.050000 TFDI_Full_pca Logit         0.039272                420                420
   12 0.050000      TFDI_pca Logit         0.092223                420                420
   12 0.050000      TFDI_avg Logit         0.086553                420                420
   12 0.100000 TFDI_Full_pca Logit         0.038544                420                420
   12 0.100000      TFDI_pca Logit         0.060814                420                420
   12 0.100000      TFDI_avg Logit         0.079707                420                420
   12 0.250000 TFDI_Full_pca Logit         0.037156                420                420
   12 0.250000      TFDI_pca Logit         0.054493                420                420
   12 0.250000      TFDI_avg Logit         0.066675                420                420


================================================================================
ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  h_qt: 3
  q_qt: 0.25
  Predictor Set: TFDI_Full_pca
  Brier Score: 0.027413

Summary by h_qt:
  h_qt=1: Average Brier Score = 0.042767
  h_qt=2: Average Brier Score = 0.043123
  h_qt=3: Average Brier Score = 0.041475

Summary by q_qt:
  q_qt=0.25: Average Brier Score = 0.037774
  q_qt=0.35: Average Brier Score = 0.042533
  q_qt=0.45: Average Brier Score = 0.047057

Summary by Predictor Set:
  TFDI_Full_pca: Average Brier Score = 0.034916
  TFDI_pca: Average Brier Score = 0.038166
  TFDI_avg: Average Brier Score = 0.054282

Detailed Results:
 h_qt     q_qt Predictor_Set Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
    1 0.250000 TFDI_Full_pca Logit         0.036162                420                420
    1 0.250000      TFDI_pca Logit         0.030631                420                420
    1 0.250000      TFDI_avg Logit         0.045625                420                420
    1 0.350000 TFDI_Full_pca Logit         0.037438                420                420
    1 0.350000      TFDI_pca Logit         0.037321                420                420
    1 0.350000      TFDI_avg Logit         0.050674                420                420
    1 0.450000 TFDI_Full_pca Logit         0.037122                420                420
    1 0.450000      TFDI_pca Logit         0.044851                420                420
    1 0.450000      TFDI_avg Logit         0.065080                420                420
    2 0.250000 TFDI_Full_pca Logit         0.036021                420                420
    2 0.250000      TFDI_pca Logit         0.035457                420                420
    2 0.250000      TFDI_avg Logit         0.049419                420                420
    2 0.350000 TFDI_Full_pca Logit         0.037626                420                420
    2 0.350000      TFDI_pca Logit         0.040254                420                420
    2 0.350000      TFDI_avg Logit         0.052505                420                420
    2 0.450000 TFDI_Full_pca Logit         0.034341                420                420
    2 0.450000      TFDI_pca Logit         0.042459                420                420
    2 0.450000      TFDI_avg Logit         0.060019                420                420
    3 0.250000 TFDI_Full_pca Logit         0.027413                420                420
    3 0.250000      TFDI_pca Logit         0.030190                420                420
    3 0.250000      TFDI_avg Logit         0.049052                420                420
    3 0.350000 TFDI_Full_pca Logit         0.033160                420                420
    3 0.350000      TFDI_pca Logit         0.040695                420                420
    3 0.350000      TFDI_avg Logit         0.053123                420                420
    3 0.450000 TFDI_Full_pca Logit         0.034957                420                420
    3 0.450000      TFDI_pca Logit         0.041640                420                420
    3 0.450000      TFDI_avg Logit         0.063042                420                420

================================================================================
ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  h_qt: 3
  q_qt: 0.25
  Predictor Set: TFDI_Full_pca
  Brier Score: 0.027499

Summary by h_qt:
  h_qt=3: Average Brier Score = 0.039821

Summary by q_qt:
  q_qt=0.1: Average Brier Score = 0.045399
  q_qt=0.15: Average Brier Score = 0.038268
  q_qt=0.2: Average Brier Score = 0.038028
  q_qt=0.25: Average Brier Score = 0.035577
  q_qt=0.3: Average Brier Score = 0.039356
  q_qt=0.35: Average Brier Score = 0.042297

Summary by Predictor Set:
  TFDI_Full_pca: Average Brier Score = 0.031948
  TFDI_pca: Average Brier Score = 0.035970
  TFDI_avg: Average Brier Score = 0.051545

Detailed Results:
 h_qt     q_qt Predictor_Set Model  Avg_Brier_Score  Valid_Predictions  Total_Predictions
    3 0.100000 TFDI_Full_pca Logit         0.037670                420                420
    3 0.100000      TFDI_pca Logit         0.042572                420                420
    3 0.100000      TFDI_avg Logit         0.055956                420                420
    3 0.150000 TFDI_Full_pca Logit         0.030333                420                420
    3 0.150000      TFDI_pca Logit         0.033443                420                420
    3 0.150000      TFDI_avg Logit         0.051027                420                420
    3 0.200000 TFDI_Full_pca Logit         0.031762                420                420
    3 0.200000      TFDI_pca Logit         0.032386                420                420
    3 0.200000      TFDI_avg Logit         0.049936                420                420
    3 0.250000 TFDI_Full_pca Logit         0.027499                420                420
    3 0.250000      TFDI_pca Logit         0.030180                420                420
    3 0.250000      TFDI_avg Logit         0.049052                420                420
    3 0.300000 TFDI_Full_pca Logit         0.031256                420                420
    3 0.300000      TFDI_pca Logit         0.036639                420                420
    3 0.300000      TFDI_avg Logit         0.050175                420                420
    3 0.350000 TFDI_Full_pca Logit         0.033166                420                420
    3 0.350000      TFDI_pca Logit         0.040601                420                420
    3 0.350000      TFDI_avg Logit         0.053123                420                420

================================================================================
FACTOR ROBUSTNESS EXERCISE RESULTS
================================================================================

Best performing combination:
  n_factors: 8
  Brier Score: 0.030107

Summary statistics:
  Mean Brier Score: 0.033404
  Std Brier Score: 0.003440
  Min Brier Score: 0.030107
  Max Brier Score: 0.043729

Performance by number of factors:
  n_factors= 1: Brier Score = 0.043729
  n_factors= 2: Brier Score = 0.033039
  n_factors= 3: Brier Score = 0.032715
  n_factors= 4: Brier Score = 0.033537
  n_factors= 5: Brier Score = 0.034032
  n_factors= 6: Brier Score = 0.032918
  n_factors= 7: Brier Score = 0.032738
  n_factors= 8: Brier Score = 0.030107
  n_factors= 9: Brier Score = 0.030544
  n_factors=10: Brier Score = 0.032186
  n_factors=11: Brier Score = 0.032312
  n_factors=12: Brier Score = 0.032995

Optimal number of factors: 8

Detailed Results:
 n_factors  Avg_Brier_Score  Valid_Predictions  Total_Predictions
         1         0.043729                420                420
         2         0.033039                420                420
         3         0.032715                420                420
         4         0.033537                420                420
         5         0.034032                420                420
         6         0.032918                420                420
         7         0.032738                420                420
         8         0.030107                420                420
         9         0.030544                420                420
        10         0.032186                420                420
        11         0.032312                420                420
        12         0.032995                420                420