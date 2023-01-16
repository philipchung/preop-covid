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

# TODO: Fix datatimes for ... (these are values from cases_df on creation--not when loading from cache)
# LastPositiveCovidInterval             object
# LabCaseIntervalCategory               object
# LabCaseIntervalCategory2              object

#%%
cases_df.LabCaseIntervalCategory.value_counts()
# Never        34307
# >=8_Weeks      424
# 0-1_Weeks      226
# 1-2_Weeks       71
# 2-3_Weeks       66
# 3-4_Weeks       64
# 4-5_Weeks       47
# 5-6_Weeks       40
# 7-8_Weeks       33
# 6-7_Weeks       25
# Name: LabCaseIntervalCategory, dtype: int64
#%%
cases_df.LabCaseIntervalCategory2.value_counts()
# >=8_Weeks      424
# 0-2_Weeks      297
# 2-4_Weeks      130
# 4-6_Weeks       87
# 6-8_Weeks       58
# Name: LabCaseIntervalCategory2, dtype: int64
#%%

# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]
# Distribution of time of last positive COVID test until Surgery Case
cases_with_positive_preop_covid.LastPositiveCovidInterval.describe()
# count                           996
# mean     75 days 19:43:16.385542169
# std      94 days 22:18:32.344486461
# min                 0 days 00:10:00
# 25%                 9 days 06:16:45
# 50%                38 days 21:15:30
# 75%               107 days 18:59:00
# max               576 days 06:24:00
# Name: LastPositiveCovidInterval, dtype: object
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

# TODO: mark new zones for 1-week and 2-week intervals

# TODO: Order of interval categories is out of order in plots--fix this

#%%
# Case Count
ax = sns.countplot(x=cases_with_positive_preop_covid.LabCaseIntervalCategory2)
ax.set(
    title="Number of Cases for each SARS-CoV-2 PCR+ Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# ASA score
ax = sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
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
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.AdmissionType,
    stat="percent",
    multiple="fill",
).set(
    title="Admission Type for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# Anesthesia Case Duration
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    y=(cases_with_positive_preop_covid.AnesDuration.dt.total_seconds() / 3600).astype(int),
).set(
    title="Anesthesia Case Duration for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Anesthesia Case Duration (hours)",
)
#%%
# PACU Duration
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory,
    y=(cases_with_positive_preop_covid.PACU_Duration.dt.total_seconds() / 60).astype(int),
).set(
    title="PACU Duration for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="PACU Duration (minutes)",
)
#%%
# Length of Stay
ax = sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory,
    y=cases_with_positive_preop_covid.Postop_LOS.dt.days,
).set(
    title="Post-op Length of Stay for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Post-op Length of Stay (days)",
)

#%%
# Count of Patients with Post-op Pulmonary Complications
ax = sns.countplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory,
    hue=cases_with_positive_preop_covid.PulmonaryComplication.apply(
        lambda x: True if "Yes" in x else False
    ),
).set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definitions) for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Count",
)
#%%
# Percent of Patients with Post-op Pulmonary Complications
ax = sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory,
    hue=cases_with_positive_preop_covid.PulmonaryComplication.apply(
        lambda x: True if "Yes" in x else False
    ),
    stat="percent",
    multiple="fill",
).set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definitions) for SARS-CoV-2 PCR+ by Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)


#%%
# TODO:
# - stratify population based on the following variable

# 3. get data on whether patient has date of COVID vaccinations (full vs booster)
#   - bool: vaccinated or not?
#   - time since last vaccination or booster?

# Outcomes
# 1. explore comorbidity index <-- export in Cases Table
# 2. PACU duration - done
# 3. Post-op Length of Stay - done
# 4. 30-day hospital mortality - done

# TODO: create only negative COVID PCR tests as comparison baseline

# %%
