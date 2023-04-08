#%%
from pathlib import Path

import numpy as np
import pandas as pd
from case_data import CaseData
from lab_data import LabData
from preop_data import PreopSDE
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

# Load & Clean Labs Data (Multiple Labs per MPOG_Case_ID)
lab_data = LabData(labs_df=labs_path, data_version=data_version)
labs_df = lab_data()

# Load & Clean Cases Data (One Row per MPOG_Case_ID)
case_data = CaseData(cases_df=cases_path, data_version=data_version)
# Associate Labs from each Patient to Cases for each Patient
cases_df = case_data.associate_labs_to_cases(labs_df=labs_df)

# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

#%%
# Load & Clean Vaccine Data (Multiple Vaccines per MPOG_Case_ID & MPOG_Patient_ID)
vaccine_data = VaccineData(vaccines_df=hm_path, data_version=data_version)
flu_vaccines_df = vaccine_data.flu_vaccines_df
covid_vaccines_df = vaccine_data.covid_vaccines_df

covid_vaccines_df.VaccineKind.value_counts()
#%%
# Load & Clean SmartDataElements Data (Multiple SDE per MPOG_Case_ID)
preop_data = PreopSDE(preop_df=preop_smartdataelements_path, data_version=data_version)
problems_df = preop_data.problems_df

problems_df.loc[problems_df.IsPresent].Problem.value_counts()

#%% [markdown]
# ## Reformat Tables so we have 1 row per MPOG_Cases_ID
# This involves aggregation of multiple rows from certain tables
# into a single row prior to joining these tables together

# %%
# Get Number of Pre-op COVID vaccines for each Case
# (row for every unique MPOG_Case_ID & Vaccine_UUID combination)
c = cases_df.reset_index().loc[:, ["MPOG_Case_ID", "MPOG_Patient_ID", "AnesStart"]]
covid_vaccines = pd.merge(
    left=c,
    right=covid_vaccines_df.reset_index(),
    how="left",
    left_on="MPOG_Patient_ID",
    right_on="MPOG_Patient_ID",
).set_index("MPOG_Case_ID")
# Only keep rows where vaccine was administered prior to Case Start
covid_vaccines = covid_vaccines.loc[covid_vaccines.AnesStart > covid_vaccines.VaccineDate]
# Aggregate Multiple Pre-op Vaccines for each MPOG_Case_ID into a list
# so we have 1 row per MPOG_Case_ID
covid_vaccines = covid_vaccines.groupby("MPOG_Case_ID")["VaccineUUID", "VaccineDate"].agg(
    {"VaccineUUID": list, "VaccineDate": list}
)
covid_vaccines["NumPreopVaccines"] = covid_vaccines.VaccineUUID.apply(len)
print(f"Num Cases with Preop Vaccine Data: {covid_vaccines.shape[0]}")

# Pivot Table for ROS Problems to get ProcID x Problems
# If any SmartDataEelement is True for a Problem (across any PreAnes Note)
# written for a specific ProcID, then we mark the Problem as True.
cases_problems = pd.pivot_table(
    data=problems_df,
    index="MPOG_Case_ID",
    columns="Problem",
    values="IsPresent",
    aggfunc=lambda x: True if any(x) else False,
    fill_value=False,
)
print(f"Num Cases with ROS Problems: {cases_problems.shape[0]}")

# Pivot Table for Organ Systems- (ProcID x OrganSystem)
# If any SmartDataElement in an OrganSystem is marked True (across any PreAnes Note)
# written for a specific ProcID, then we mark the OrganSystem as True.
cases_organ_systems = pd.pivot_table(
    data=problems_df,
    index="MPOG_Case_ID",
    columns="OrganSystem",
    values="IsPresent",
    aggfunc=lambda x: True if any(x) else False,
    fill_value=False,
)
print(f"Num Cases with ROS Problems Marked by Organ Systems: {cases_organ_systems.shape[0]}")
#%%
# Join Vaccines, SDE Problems & Organ Systems Data to original cases table
# Note: original cases table has all the Elixhauser Comorbidities & Complications
# as well as time between last PCR+ test and case
# We drop cases where we don't have SDE data

# Join Vaccines (MPOG_Case_ID without vaccines are unvaccinated)
df = cases_df.join(covid_vaccines, how="left")

df.VaccineUUID = [[] if x is np.NaN else x for x in df.VaccineUUID]
df.VaccineDate = [[] if x is np.NaN else x for x in df.VaccineDate]
df.NumPreopVaccines = df.NumPreopVaccines.fillna(0)

# Join SDE Data
df = df.join(cases_organ_systems, how="inner").join(cases_problems, how="inner")
print(f"Num Cases: {df.shape[0]}")

# Preview this Table
df

#%%
# Covid Vaccine Columns
covid_vaccines.columns
#%%
# All ROS Problems
cases_problems.columns
#%%
# All ROS Organ Systems
cases_organ_systems.columns
#%%
# Case Data (Elixhauser Comorbidiites, Complications, PCR Data)
cases_df.columns

#%% [markdown]
# Now we can interrogate almost any patient population based on combination of ROS
# or COVID Vaccination Status and look at the MPOG documented complication
# or Elixhauser Comorbidities
# NOTE:
# - Any Cohort selection is based on ROS Data from Pre-op Note
# - Any Outcome Complications or Elixhauser Comorbidities is based on case billing data
# (e.g. ICD codes)

#%%
# Cohort = Patients with COPD (in ROS)
copd = df.loc[df.COPD]
# Outcome = PulmonaryComplication (by case ICD code)
copd.HadPulmonaryComplication.value_counts()
#%%
# Outcome = CardiacComplication (by case ICD code)
copd.HadCardiacComplication.value_counts()
#%%
# Outcome = AKIComplication (by case ICD code)
copd.HadAKIComplication.value_counts()
#%%
# Cohort = Patients with COPD or Asthma or Bronchitis or URI (in ROS)
acute_or_chronic_pulm_dz = df.loc[df.COPD | df.ASTHMA | df.BRONCHITIS | df.URI]
# Outcome = PulmonaryComplication (by case ICD code)
acute_or_chronic_pulm_dz.HadPulmonaryComplication.value_counts()
#%%
# Outcome = CardiacComplication (by case ICD code)
acute_or_chronic_pulm_dz.HadCardiacComplication.value_counts()
#%%
# Outcome = AKIComplication (by case ICD code)
acute_or_chronic_pulm_dz.HadAKIComplication.value_counts()
#%%
# cohort = Patient with any +Respiratory ROS
any_respiratory = df.loc[df.RESPIRATORY]
# Outcome = PulmonaryComplication (by case ICD code)
any_respiratory.HadPulmonaryComplication.value_counts()
#%%
# Outcome = CardiacComplication (by case ICD code)
any_respiratory.HadCardiacComplication.value_counts()
#%%
# Outcome = AKIComplication (by case ICD code)
any_respiratory.HadAKIComplication.value_counts()
#%%
# cohort = Patient with any +Respiratory ROS
any_respiratory.loc[:, ["HadPulmonaryComplication", "NumPreopVaccines"]].value_counts()

#
#%%
# TODO:
# 1. pick out top diagnoses (CV, pulm, renal) from ROS/problems_df that we want to examine
# 2. join ROS/problems into cases_df table (alongside Elixhauser Comorbidities)
#    so each case has associated ROS (binary values)
# 3. for each case, get total # of vaccines/boosters; get # of vaccines/booster & duration
#    since last vaccine/booster administration.
# 4. For patients with vaccine in past 6 months, compare outcomes (PACU duration, Hospital LOS,
#    Mortality)
# 5. repeat for patients with vaccine in past 12 months...
# (NEJM paper suggests clinical protection for 6 months: https://www.nejm.org/doi/full/10.1056/NEJMoa2118691)
# 6. Patients with Heart Disease
# %%
# Call w/ Dustin Notes
# - self antigen tests
# - walgreens?  most pre-op is being done at local provider
# - free-text comment
# (maybe just secondary analysis)

# limit to possible covid?  home test?
# look just at vaccination status.  Difference in all-comers.

# - is there a difference risk between 2-3 post-op complications between vaccinated & non-vaccinated

# ROS groups:
# - COPD (assoc. w/ COVID)
# - CHF (interaction w/ COVID)
# - h/o MI
# - stroke (inc. risk w/ stroke w/ COVID)
