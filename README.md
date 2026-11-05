# Delhivery Last-Mile Delay Intelligence

An offline, reproducible pipeline for shipment preparation, delivery-time regression, delay classification, hub segmentation, and dispatch-time actions. The shipment records are synthetic and do not represent Delhivery's real operations.

## How to run

Run these commands from the repository root in order:

```powershell
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py

```

`generate_data.py` creates the source CSVs; Part A creates `data/modelling_table.csv`; the analysis scripts write their evidence to `outputs/`. `package_model.py` regenerates `models/delay_model.pkl`, the stored threshold, feature order, and imputation medians. A committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; when installed versions differ, the grader should rerun `python package_model.py` instead of loading the committed pickle.

## Data findings

The cleaned table has 1200 unique historical shipments; Express has the highest observed delay rate at 0.4897. The rate rises from Tier 1 0.0400 to Tier 3 0.4400. Across distance bands the rates are 0-200 km 0.0219, 201-400 km 0.1368, 401-600 km 0.4364, 601+ km 0.8523. These values are computed in `outputs/part_a_delay_by_group.csv`.

The three strongest signed correlations with delivery hours are promised_hours +0.8372, distance_km +0.4013, n_stops +0.3247 (`outputs/part_a_correlations.csv`). Filled declared value has mean INR 1533.05 and median INR 1121.50; after log transformation the mean is 7.0277 and median 7.0224. The log compresses the right tail so high declared values exert less leverage.

## Model results (regression)

The mean baseline has validation R-squared -0.0212; a constant prediction is weak because it cannot account for service promises, route length, and hub conditions. The validation-RMSE winner is linear_regression at 4.8912 hours. At depth 1, train/validation RMSE are 11.6334/12.1324, consistent with high bias; an unlimited tree has 0.0000/7.8368, showing the train-validation gap associated with high variance. The best gradient-boosting learning rate is 0.10 (validation R-squared 0.9306); 0.01 learns too slowly, while 1.00 overshoots the useful step size. Cross-validated R-squared ranks are gradient_boosting 0.9353 (SD 0.0045), linear_regression 0.9261 (SD 0.0048). The CV ranking differs from validation; the validation winner is retained because it is selected by the prescribed validation RMSE rule, while CV is used as a robustness check. See `outputs/part_b_metrics.csv`, `outputs/part_b_tree_depth.csv`, `outputs/part_b_gb_learning_rate.csv`, and `outputs/part_b_cv.csv`.

## Model results (classification)

The all-on-time baseline has accuracy 0.7667, but recall is 0.0000: it detects no delayed parcels. Noise columns account for 0.2245 of the default forest's importance; all five rank outside the top seven by F-score: True. The best validation F1 in the `k` comparison is 0.8333 at k=7. Tuning F1 is grid 0.7750, randomized 0.7848, and untuned 0.8333; tuning does not beat the untuned forest on this 180-row validation set, so a small difference should not be treated as strong evidence of generalization.

The selected source is `default_forest_same_7_features` with threshold 0.5. At that threshold validation precision is 0.7955 and recall is 0.8333, the highest threshold meeting the 0.80 recall rule. On test data, precision is 0.6889, recall 0.7561, and F1 0.7209; test recall is lower than validation recall. The test set contains only 41 delayed parcels, so a few outcomes materially affect the comparison. See the Part C CSVs under `outputs/`.

## Operations recommendation

The Escalate hubs and their observed delay rates are HUB13 0.4600; HUB15 0.3600; HUB17 0.4600; HUB18 0.3800; HUB19 0.6200; HUB20 0.4800; HUB21 0.4000; HUB23 0.4400; HUB24 0.5400. Tier-versus-segment mismatches are HUB07 Kolkata (Tier 1, Watch); HUB13 Nagpur (Tier 2, Escalate); HUB15 Kochi (Tier 2, Escalate); HUB22 Dehradun (Tier 3, Watch); use each hub's observed segment to qualify its tier label rather than treating the tier as ground truth. The alert threshold is 0.5; test precision means 6.9 of every 10 flagged parcels were delayed, and test recall means 7.6 of every 10 delayed parcels were caught. Of the 48 new shipments, 8 are flagged and 6 are routed to a hub manager.

Keep K=3 because it matches the three available playbooks and produces operationally sized groups. The across-to-within ratio increases from 5.393 at K=3 to 6.305 at K=4 and 6.887 at K=5, but K=4 sizes are 8-7-5-4 and K=5 sizes are 8-7-4-3-2; both include clusters smaller than six hubs. K=2 has sizes 15-9 and ratio 2.894. See `outputs/part_d_separation.csv`.

## Production plan

- **Accuracy:** validation and held-out test metrics are kept separate in the Part B and Part C outputs.
- **Reliability:** `part_d_reload_check.csv` verifies that the serialized pipeline reproduces in-memory probabilities.
- **Reproducibility:** fixed random seeds, pinned package versions, deterministic input generation, and index-free CSV outputs make each run repeatable.
- **Maintainability:** shared feature definitions and transformations live in `src/shipment_prep.py` instead of being copied into model and scoring scripts.
- **Preprocessing consistency:** `predict_delay.py` calls the same `clean_shipments` and `build_features` functions used during training and uses the saved training medians and feature order.
- **Monitorability:** retain weekly delayed share, flagged share, and recall on parcels whose outcomes have matured. Investigate if delayed share changes by more than 5 percentage points from the same seasonal/event baseline, flagged share changes by more than 5 percentage points, or matured-parcel recall falls below 0.70.
- **Performance:** the batch scorer uses a single fitted forest pipeline and scores 48 rows locally without a paid service.

Retrain quarterly, and review sooner after sustained drift or a material operating change. During the 15-28 September sale window, compare `sale_event_flag=1` parcels with the prior sale window rather than ordinary weeks. Drift would appear as sustained changes in distance, stops, hub mix, sale-event share, predicted probabilities, or delayed share relative to the relevant seasonal baseline.

## Limitations

- The dataset is synthetic and cannot establish performance on real shipment workflows or real hub behavior.
- The test set contains only 41 delayed parcels, so a small number of outcomes can materially change recall and F1.
- The threshold and model choice are based on one fixed validation split; repeated temporal or cross-hub validation could reveal less favorable generalization.
- Hub segments are descriptive clusters of only 24 hubs and should not be interpreted as causal explanations.
