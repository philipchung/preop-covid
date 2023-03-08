#%%
from pathlib import Path

import pandas as pd
from case_data import CaseData
from lab_data import LabData
from preop_data import PreopData
from vaccine_data import VaccineData

# Define data paths and directories
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
data_version = 2
raw_dir = data_dir / f"v{int(2)}" / "raw"
processed_dir = data_dir / f"v{int(2)}" / "processed"
[path.mkdir(parents=True, exist_ok=True) for path in (raw_dir, processed_dir)]  # type: ignore
cohort_details_path = raw_dir / "Cohort Details_9602.csv"
summary_path = raw_dir / "Summary_9602.csv"
cases_path = raw_dir / "Main_Case_9602.csv"
labs_path = raw_dir / "SF_309 Covid Labs.csv"
preop_smartdataelements_path = raw_dir / "CF_2023-02-27_ChungCOVIDPreAnesEval.csv"
hm_path = raw_dir / "CF_2023-02-27_ChungCOVIDHealthMaintenance.csv"
#%%
# Load Metadata
raw_cohort_df = pd.read_csv(cohort_details_path)
raw_summary_df = pd.read_csv(summary_path)

# Load & Clean Labs Data
lab_data = LabData(labs_df=labs_path, data_version=data_version)
labs_df = lab_data()

# Load & Clean Cases Data
case_data = CaseData(cases_df=cases_path, data_version=data_version)
cases_df = case_data()

# Associate Labs from each Patient to Cases for each Patient
case_data.associate_labs_to_cases(labs_df=labs_df, update_cases_df=True)

# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

# Load & Clean Vaccine Data
vaccine_data = VaccineData(vaccines_df=hm_path, data_version=data_version)

#%%
# Load & Clean SmartDataElements Data
preop_data = PreopData(preop_df=preop_smartdataelements_path, data_version=data_version)
problems_df = preop_data.problems_df

#%%
