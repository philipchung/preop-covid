#%%
from __future__ import annotations

from pathlib import Path

import pandas as pd
from case_data import CaseData
from lab_data import LabData

# Define data paths and directories
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
raw_dir = data_dir / "v1.1" / "raw"
processed_dir = data_dir / "v1.1" / "processed"
[path.mkdir(parents=True, exist_ok=True) for path in (raw_dir, processed_dir)]  # type: ignore
cohort_details_path = raw_dir / "Cohort Details_9602.csv"
summary_path = raw_dir / "Summary_9602.csv"
cases_path = raw_dir / "Results_Main_Case_9602.csv"
diagnosis_path = raw_dir / "Results_DiagnosesCleanedAggregated_9602.csv"
labs_path = raw_dir / "SF_309 Covid Labs.csv"
ros_path = raw_dir / "CF_2023-02-24_ChungCOVIDReviewOfSystems.csv"
hm_path = raw_dir / "CF_2023-02-24_ChungCOVIDHealthMaintenance.csv"
#%%
# Load Metadata
raw_cohort_df = pd.read_csv(cohort_details_path)
raw_diagnosis_df = pd.read_csv(diagnosis_path)
raw_summary_df = pd.read_csv(summary_path)

# Load & Clean Labs Data
lab_data = LabData(labs_df=labs_path)
labs_df = lab_data()

# Load & Clean Cases Data
case_data = CaseData(cases_df=cases_path)
case_data.associate_labs_to_cases(labs_df=labs_df)
cases_df = case_data()
# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

#%%
# Get ROS Table
ros_df = pd.read_csv(ros_path, encoding="latin1")
# Get Health Maintenance / Vaccine Table
hm_df = pd.read_csv(hm_path)
# %%
