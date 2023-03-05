#%%
from pathlib import Path

from vaccine_data import VaccineData

# Define data paths and directories
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
data_version = 2.0
raw_dir = data_dir / f"v{int(2)}" / "raw"
processed_dir = data_dir / f"v{int(2)}" / "processed"
[path.mkdir(parents=True, exist_ok=True) for path in (raw_dir, processed_dir)]  # type: ignore
cohort_details_path = raw_dir / "Cohort Details_9602.csv"
summary_path = raw_dir / "Summary_9602.csv"
cases_path = raw_dir / "Main_Case_9602.csv"
labs_path = raw_dir / "SF_309 Covid Labs.csv"
ros_path = raw_dir / "CF_2023-02-27_ChungCOVIDPreAnesEval.csv"
hm_path = raw_dir / "CF_2023-02-27_ChungCOVIDHealthMaintenance.csv"
#%%
# # Load Metadata
# raw_cohort_df = pd.read_csv(cohort_details_path)
# raw_summary_df = pd.read_csv(summary_path)

# # Load & Clean Labs Data
# lab_data = LabData(labs_df=labs_path, data_version=data_version)
# labs_df = lab_data()

# # Load & Clean Cases Data
# case_data = CaseData(cases_df=cases_path, data_version=data_version)
# cases_df = case_data()

# # Associate Labs from each Patient to Cases for each Patient
# case_data.associate_labs_to_cases(labs_df=labs_df, update_cases_df=True)

# # Get only patients with a positive Preop COVID test
# cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

# #%%
# # Load ROS Table
# ros_df = pd.read_csv(ros_path)
#%%
# Load Health Maintenance / Vaccine Table
vaccine_data = VaccineData(vaccines_df=hm_path)

# TODO: cache result for instant load
#%%
# %%
# %%
ros_df.SmartDataElement.value_counts()
# WORKFLOW - ROS - NEGATIVE SKIN ROS                                            24445
# WORKFLOW - ROS - NEGATIVE HEMATOLOGY/ONCOLOGY ROS                             21323
# WORKFLOW - ROS - NEGATIVE HEMATOLOGY ROS                                      21078
# WORKFLOW - ROS - NEG ENDO/OTHER ROS                                           20647
# WORKFLOW - ROS - NEG PULMONARY ROS                                            19446
# WORKFLOW - ROS - NEGATIVE HEENT ROS                                           17092
# WORKFLOW - ROS - NEG CARDIO ROS                                               16457
# WORKFLOW - ROS - ROS RESPIRATORY COMMENTS                                     16414
# WORKFLOW - ROS - ROS CARDIO COMMENTS                                          16372
# WORKFLOW - ROS - NEGATIVE MUSCULOSKELETAL ROS                                 15336
# WORKFLOW - ROS - NEG NEURO/PSYCH ROS                                          14818
# WORKFLOW - ROS - ROS RENAL COMMENTS                                           14626
# WORKFLOW - ROS - ROS MUSCULOSKELETAL COMMENTS                                 13475
# WORKFLOW - ROS - NEG GI/HEPATIC/RENAL ROS                                     13385
# WORKFLOW - ROS - NEURO/PSYCH TITLE - COMMENTS                                 13271
# WORKFLOW - ROS - ROS HEENT COMMENTS                                            7823
# WORKFLOW - ROS - ROS DERMATOLOGICAL/IMMUNOLOGICAL/RHEUMATOLOGICAL COMMENTS     5809
# WORKFLOW - ROS - ENDO/OTHER TITLE - COMMENTS                                   5682
# WORKFLOW - ROS - ONCOLOGY COMMENTS                                             5221
# WORKFLOW - ROS - ROS HEMATOLOGY COMMENTS                                       3853
# DIAGNOSES/PROBLEMS - NEUROLOGICAL - MULTIPLE SCLEROSIS                          170
# WORKFLOW - ROS - OB ROS COMMENT                                                 108
# DIAGNOSES/PROBLEMS - OBSTETRIC - PRENATAL COMPLICATIONS - MACROSOMIA             26
# WORKFLOW - ROS - NEG OB ROS                                                      24
# %%
ros_df.SmartDataAbbrev.value_counts()
# negative skin ROS                                            24445
# negative hematology/oncology ROS                             21323
# negative hematology ROS                                      21078
# neg endo/other ROS                                           20647
# neg pulmonary ROS                                            19446
# negative HEENT ROS                                           17092
# neg cardio ROS                                               16457
# ROS respiratory comments                                     16414
# ROS cardio comments                                          16372
# negative musculoskeletal ROS                                 15336
# neg neuro/psych ROS                                          14818
# ROS renal comments                                           14626
# ROS musculoskeletal comments                                 13475
# neg GI/hepatic/renal ROS                                     13385
# neuro/psych title - comments                                 13271
# ROS HEENT comments                                            7823
# ROS dermatological/immunological/rheumatological comments     5809
# Endo/other title - comments                                   5682
# oncology comments                                             5221
# ROS hematology comments                                       3853
# multiple sclerosis                                             170
# OB ROS Comment                                                 108
# macrosomia                                                      26
# neg OB ROS                                                      24
# ROS GI comments                                                  1
# %%
ros_df.SmartElemValue.value_counts()
# There are 80089 unique values.
# %%
mpog_case_id, case_ros_df = next(iter(ros_df.groupby("MPOG_Case_ID")))
# %%
case_ros_df
# %%
# Notes:
# - "neg OB ROS" is a check box for L&D C-section patients only under the "OB/GYN" tab
# - OB ROS Comment is the comment box for entire "Obstetric" part under "OB/GYN tab"
# - We should drop DIAGNOSES/PROBLEMS & just focus on WORKFLOW... these might be a contaminant?
#   - unclear where multiple sclerosis is documented
#   - Prenatal Complications - Macrosomia is under "OB/GYN" tab as part of "Obstetric"
# -
#%%
