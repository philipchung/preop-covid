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
cases_df = case_data()

#%%
mpog_patient_id, case_df = next(iter(case_data.cases_df.groupby("MPOG_Patient_ID")))
labs = labs_df.loc[labs_df.MPOG_Patient_ID == mpog_patient_id].sort_values(
    by="DateTime", ascending=True
)
case = case_df.iloc[0, :]
res = case_data.last_covid_lab_before_case(
    labs_df=labs, mpog_patient_id=mpog_patient_id, case_start=case.AnesStart
)

#%%
# Associate Labs from each Patient to Cases for each PAtient
case_data.associate_labs_to_cases(labs_df=labs_df)
cases_df = case_data()
# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

#%%
# Load ROS Table
ros_df = pd.read_csv(ros_path, encoding="latin1")
# Load Health Maintenance / Vaccine Table
hm_df = pd.read_csv(hm_path)
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
# Explore health maintenance
hm_df.HMTopic.value_counts()
# Influenza Vaccine                       252462
# COVID-19 Vaccine                        122733
# Influenza (inactivated) (Sept-March)     99860
# Haemophilus Influenzae type B             8087
# ZZZINFLUENZA VACCINE                      6199
# ZZZOB Influenza Vaccine                   5448
# ZZZINFLUENZA >9 YRS                       1375
# ZZZOB-INFLUENZA VACCINE,REVISED            970
# Name: HMTopic, dtype: int64

hm_df.HMType.value_counts()
# Done                    495105
# Declined                  1576
# Previously completed       117
# Not indicated               42
# Name: HMType, dtype: int64

# %%
import yaml
from pandas.api.types import CategoricalDtype

vaccine_category = CategoricalDtype(categories=["COVID-19", "Influenza", "Other"], ordered=False)

health_maintenance_values_map_path: str | Path = (
    Path(__file__).parent / "health_maintenance_values_map.yml"
)
health_maintenance_values_map = yaml.safe_load(Path(health_maintenance_values_map_path).read_text())


def clean_hm_topic_value(value: str) -> str:
    """Converts health maintenance topic field into Categorical value.

    Args:
        value (str): string value for HMTopic field

    Returns:
        str: "COVID-19", "Influenza", "Other"
    """
    value = value.lower().strip()
    if value in [x.lower() for x in health_maintenance_values_map["covid-19_vaccine"]]:
        return "COVID-19"
    elif value in [x.lower() for x in health_maintenance_values_map["influenza_vaccine"]]:
        return "Influenza"
    elif value in [x.lower() for x in health_maintenance_values_map["other"]]:
        return "Other"
    else:
        raise ValueError(
            f"Unknown value {value} encountered that is not handled by "
            "clean_hp_topic_value() logic."
        )


hm_df = pd.read_csv(hm_path)
res = hm_df.loc[hm_df.HMType == "Done"]
res["VaccineType"] = res.HMTopic.apply(clean_hm_topic_value).astype(vaccine_category)
res = res.loc[res.VaccineType != "Other"]
res
#%%
# TODO:
# - parse comments -- need to do separately for COVID-19 & Flu
res_covid = res.loc[res.VaccineType == "COVID-19"]
res_flu = res.loc[res.VaccineType == "Influenza"]
# %%
res_covid.COMMENTS.head().tolist()
# ['Imm Admin: LIMId-153^LPLId-19924636^\x11COVID-19 Janssen vector-nr rS-Ad26',
#  'Imm Admin: LIMId-143^LPLId-26356319^\x11COVID-19 Moderna mRNA 12 yrs and older',
#  'Imm Admin: LIMId-179^LPLId-25200332^\x11COVID-19 Pfizer mRNA tris-sucrose 12 yrs and older (gray cap)',
#  'Imm Admin: LIMId-143^LPLId-31600302^\x11COVID-19 Moderna mRNA 12 yrs and older',
#  'Imm Admin: LIMId-144^LPLId-21406720^\x11COVID-19 Pfizer mRNA 12 yrs and older (purple cap)']
#%%
res_covid.COMMENTS.tail().tolist()
# ['Ext Imm: LIMID-144^\x11COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'Ext Imm: LIMID-144^\x11COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'Ext Imm: LIMID-144^\x11COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'Ext Imm: LIMID-144^\x11COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'Ext Imm: LIMID-190^\x11COVID-19, mRNA, LNP-S, bivalent booster, PF, 50 mcg/0.5 mL or 25mcg/0.25 mL dose\x11\x110']
# %%
