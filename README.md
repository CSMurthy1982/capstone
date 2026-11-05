# capstone
# capstone project Delhivery Last-Mile Delay Intelligence

Run these commands from the repository root in order:

pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
python write_readme.py

"generate_data.py" creates the source CSVs

Part A creates "data/modelling_table.csv"; 
the analysis scripts write their evidence to "outputs/". "package_model.py` regenerates `models/delay_model.pkl`,
the stored threshold, feature order, and imputation medians. 

A committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; when installed versions differ, the grader should rerun `python package_model.py` instead of loading the committed pickle.


