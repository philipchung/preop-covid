#%%
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from case_data import CaseData
from lab_data import LabData
from matplotlib import pyplot as plt

# Define data paths and directories
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
raw_dir = data_dir / "v1" / "raw"
processed_dir = data_dir / "v1" / "processed"
[path.mkdir(parents=True, exist_ok=True) for path in (raw_dir, processed_dir)]  # type: ignore
cohort_details_path = raw_dir / "Cohort Details_9602.csv"
diagnosis_path = raw_dir / "Results_DiagnosesCleanedAggregated_9602.csv"
labs_path = raw_dir / "Results_Laboratories_9602.csv"
cases_path = raw_dir / "Results_Main_Case_9602.csv"
summary_path = raw_dir / "Summary_9602.csv"
#%%
# Load Metadata
raw_cohort_df = pd.read_csv(cohort_details_path)
raw_diagnosis_df = pd.read_csv(diagnosis_path)
raw_summary_df = pd.read_csv(summary_path)

# Load Labs & Cases Data, Clean & Process
lab_data = LabData(labs_df=labs_path)
labs_df = lab_data()

case_data = CaseData(cases_df=cases_path)
case_data.associate_labs_to_cases(labs_df=labs_df)
cases_df = case_data()
#%%


# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df.loc[cases_df.HasPositivePreopCovidTest]
# Categorical Groups
cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory.value_counts().sort_index()
# 0-2_weeks    383
# 3-4_weeks    116
# 5-6_weeks     70
# >=7_weeks    467
# Name: LastPositivePreopCovidIntervalCategory, dtype: int64
#%%
# Distribution of time of last positive COVID test until Surgery Case
cases_with_positive_preop_covid.LastPositivePreopCovidInterval.describe()
# count                          1036
# mean     74 days 13:39:48.011583012
# std      94 days 01:45:20.751893201
# min                 0 days 00:10:00
# 25%                 8 days 14:09:30
# 50%                37 days 23:00:00
# 75%               105 days 17:35:30
# max               576 days 06:24:00
# Name: LastPositivePreopCovidInterval, dtype: object
# %%
# Swarm Plot Showing COVID+ Patients and time Interval Until Surgery
# Colored regions indicate time intervals form categories

fig = plt.figure(figsize=(10, 5))
p1 = sns.swarmplot(
    data=cases_with_positive_preop_covid.LastPositivePreopCovidInterval.dt.days, orient="h", size=3
)
p1.set(
    xlabel="Number of Days Since Positive SARS-CoV-2 PCR Test",
    title="Time Interval Between PCR-confirmed COVID19 Infection and Surgery",
)
# Orange = 0-2 weeks (actually 0-2.5 weeks)
p1.fill_betweenx(
    y=[-0.4, 0.4], x1=0, x2=17, alpha=0.5, color=matplotlib.colors.TABLEAU_COLORS["tab:orange"]
)
# Red = 3-4 weeks (actually 2.5-4.5 weeks)
p1.fill_betweenx(
    y=[-0.4, 0.4], x1=17, x2=31, alpha=0.5, color=matplotlib.colors.TABLEAU_COLORS["tab:red"]
)
# Cyan = 5-6 weeks (actually 4.5-7.5 weeks)
p1.fill_betweenx(
    y=[-0.4, 0.4], x1=31, x2=45, alpha=0.5, color=matplotlib.colors.TABLEAU_COLORS["tab:cyan"]
)
# White: >= 7 weeks (actually 7.5+ weeks)
#%%
# Case Count
ax = sns.countplot(x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory)
ax.set(
    title="Number of Cases for each SARS-CoV-2 PCR+ Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# ASA score
ax = sns.histplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    hue=cases_with_positive_preop_covid.ASA,
    stat="percent",
    multiple="fill",
)
ax.set(
    title="ASA for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# Admission Status
ax = sns.histplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    hue=cases_with_positive_preop_covid.AdmissionType,
    stat="percent",
    multiple="fill",
)
ax.set(
    title="ASA for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# Anesthesia Case Duration
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    y=(cases_with_positive_preop_covid.AnesDuration.dt.total_seconds() / 3600).astype(int),
)
ax.set(
    title="Anesthesia Case Duration for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Anesthesia Case Duration (hours)",
)
#%%
# PACU Duration
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    y=(cases_with_positive_preop_covid.PACU_Duration.dt.total_seconds() / 60).astype(int),
)
ax.set(
    title="PACU Duration for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="PACU Duration (minutes)",
)
#%%
# Length of Stay
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    y=cases_with_positive_preop_covid.Postop_LOS.dt.days,
)
ax.set(
    title="Post-op Length of Stay for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Post-op Length of Stay (days)",
)

#%%
#%%
# Count of Patients with Post-op Pulmonary Complications
ax = sns.countplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    hue=cases_with_positive_preop_covid.PulmonaryComplication.apply(
        lambda x: True if "Yes" in x else False
    ),
)
ax.set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definitions) for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Count",
)
#%%
# Percent of Patients with Post-op Pulmonary Complications
ax = sns.histplot(
    x=cases_with_positive_preop_covid.LastPositivePreopCovidIntervalCategory,
    hue=cases_with_positive_preop_covid.PulmonaryComplication.apply(
        lambda x: True if "Yes" in x else False
    ),
    stat="percent",
    multiple="fill",
)
ax.set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definitions) for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)


#%%

# TODO:
# - stratify population based on the following variable
# Predictors

# 3. get data on whether patient has date of COVID vaccinations (full vs booster)
#   - bool: vaccinated or not?
#   - time since last vaccination or booster?
# 4. demographics, age, sex, ASA score?

# Outcomes
# 1. explore comorbidity index
# 2. PACU duration
# 3. Post-op Length of Stay
# 4. 30-day hospital mortality

# %%
