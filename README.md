# capstone project : Delhivery Last-Mile Delay Intelligence

Run these commands from the repository root in order:

```
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
python write_readme.py
```

"generate_data.py" creates the source CSVs; 

Part A creates "data/modelling_table.csv"; 

the analysis scripts write their evidence to "outputs/". 

"package_model.py" regenerates "models/delay_model.pkl", the stored threshold, feature order, and imputation medians. 

A committed pickle loads reliably only with the scikit-learn version pinned in "requirements.txt"; 

when installed versions differ, the grader should rerun "python package_model.py" instead of loading the committed pickle.


## Part A - Data findings

The cleaned table has 1200 unique historical shipments; 

Express has the highest observed delay rate at 0.4897. 

The rate rises from Tier 1 0.0400 to Tier 3 0.4400. 

Across distance bands the rates are 0-200 km 0.0219, 201-400 km 0.1368, 401-600 km 0.4364, 601+ km 0.8523. 

These values are computed in "outputs/part_a_delay_by_group.csv".

The three strongest signed correlations with delivery hours are promised_hours +0.8372, distance_km +0.4013, n_stops +0.3247 ("outputs/part_a_correlations.csv"). 

Filled declared value has mean INR 1533.05 and median INR 1121.50; after log transformation the mean is 7.0277 and median 7.0224. The log compresses the right tail so high declared values exert less leverage.

## Part B -  Model results (regression)

The mean baseline has validation R-squared -0.0212; 

a constant prediction is weak because it cannot account for service promises, route length, and hub conditions. 

The validation-RMSE winner is linear_regression at 4.8912 hours. 

At depth 1, train/validation RMSE are 11.6334/12.1324, consistent with high bias; 

an unlimited tree has 0.0000/7.8368, showing the train-validation gap associated with high variance. 

The best gradient-boosting learning rate is 0.10 (validation R-squared 0.9306);

 0.01 learns too slowly, while 1.00 overshoots the useful step size.

 Cross-validated R-squared ranks are gradient_boosting 0.9353 (SD 0.0045), linear_regression 0.9261 (SD 0.0048). 
 
 The CV ranking differs from validation; the validation winner is retained because it is selected by the prescribed validation RMSE rule, while CV is used as a robustness check. 
 
 See "outputs/part_b_metrics.csv", "outputs/part_b_tree_depth.csv", "outputs/part_b_gb_learning_rate.csv", and "outputs/part_b_cv.csv".
