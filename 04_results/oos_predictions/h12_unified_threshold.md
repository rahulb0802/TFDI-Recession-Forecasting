|index|Predictor\_Set|Model|ROC\_AUC|Brier\_Score|Resolution|Reliability|Uncertainty|Num\_Forecasts|
|---|---|---|---|---|---|---|---|---|
|27|Deter\_PCA|XGBoost|0\.7193|0\.0836|0\.0128|0\.0109|0\.0862|420|
|11|PCA\_Factors\_8|XGBoost|0\.7645|0\.1012|0\.0104|0\.0264|0\.0862|420|
|19|Deter|XGBoost|0\.7341|0\.1052|0\.0085|0\.0288|0\.0862|420|
|31|Deter\_Avg|XGBoost|0\.661|0\.1096|0\.0052|0\.0291|0\.0862|420|
|23|Deter\_States|XGBoost|0\.7117|0\.1106|0\.0103|0\.0352|0\.0862|420|
|8|PCA\_Factors\_8|Logit|0\.8618|0\.1135|0\.0185|0\.0456|0\.0862|420|
|7|Full|XGBoost|0\.789|0\.12|0\.0079|0\.0432|0\.0862|420|
|17|Deter|Logit\_L1|0\.8502|0\.1323|0\.0157|0\.0627|0\.0862|420|
|21|Deter\_States|Logit\_L1|0\.7588|0\.1365|0\.0099|0\.0599|0\.0862|420|
|22|Deter\_States|Logit\_L2|0\.8024|0\.1367|0\.0132|0\.0628|0\.0862|420|
|18|Deter|Logit\_L2|0\.8535|0\.1399|0\.0155|0\.0701|0\.0862|420|
|24|Deter\_PCA|Logit|0\.7366|0\.1405|0\.0115|0\.0645|0\.0862|420|
|1|Yield|Logit\_L1|0\.8047|0\.1416|0\.015|0\.0706|0\.0862|420|
|16|Deter|Logit|0\.7455|0\.144|0\.01|0\.0674|0\.0862|420|
|0|Yield|Logit|0\.8055|0\.1459|0\.0174|0\.0764|0\.0862|420|
|10|PCA\_Factors\_8|Logit\_L2|0\.8695|0\.146|0\.0165|0\.0775|0\.0862|420|
|2|Yield|Logit\_L2|0\.8069|0\.1483|0\.0155|0\.077|0\.0862|420|
|3|Yield|XGBoost|0\.7076|0\.1491|0\.0054|0\.0699|0\.0862|420|
|26|Deter\_PCA|Logit\_L2|0\.8082|0\.1519|0\.0111|0\.0777|0\.0862|420|
|6|Full|Logit\_L2|0\.8451|0\.1622|0\.0112|0\.0888|0\.0862|420|
|25|Deter\_PCA|Logit\_L1|0\.8219|0\.1807|0\.0132|0\.1072|0\.0862|420|
|28|Deter\_Avg|Logit|0\.6472|0\.1866|0\.0049|0\.1044|0\.0862|420|
|9|PCA\_Factors\_8|Logit\_L1|0\.8285|0\.1947|0\.0092|0\.1198|0\.0862|420|
|30|Deter\_Avg|Logit\_L2|0\.6684|0\.2095|0\.0024|0\.1258|0\.0862|420|
|20|Deter\_States|Logit|0\.6878|0\.2181|0\.0038|0\.1357|0\.0862|420|
|15|ADS|XGBoost|0\.5255|0\.2263|0\.0009|0\.1416|0\.0862|420|
|29|Deter\_Avg|Logit\_L1|0\.6742|0\.2285|0\.0044|0\.1456|0\.0862|420|
|5|Full|Logit\_L1|0\.5|0\.25|0\.0|0\.1638|0\.0862|420|
|13|ADS|Logit\_L1|0\.6218|0\.2573|0\.0013|0\.1727|0\.0862|420|
|14|ADS|Logit\_L2|0\.6474|0\.2578|0\.0023|0\.1743|0\.0862|420|
|12|ADS|Logit|0\.6473|0\.258|0\.0023|0\.1745|0\.0862|420|
|4|Full|Logit|0\.6389|0\.3432|0\.0028|0\.26|0\.0862|420|

**Takeaways**
- While it seems like `Deter_PCA` + XGBoost does good here, its recall is significantly low (9/40 recession months correct)
- The models with higher AUC generally do better here rather than Brier Score
- Other than PCA on $X_t$, supervised aggregation scheme provides a good balance between AUC and Brier Score
- Yield curve + Logit has very good recall (29/40 recession months right) but at the cost of false positives (300/380 nonrecession)
