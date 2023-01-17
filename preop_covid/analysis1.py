#%%
from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from case_data import CaseData
from lab_data import LabData
from matplotlib import colors
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
# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]
#%% [markdown]
# ## Lab Case Interval Definitions
#%%
# 1-week Intervals
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
# 2-week Intervals
cases_df.LabCaseIntervalCategory2.value_counts()
# >=8_Weeks      424
# 0-2_Weeks      297
# 2-4_Weeks      130
# 4-6_Weeks       87
# 6-8_Weeks       58
# Name: LabCaseIntervalCategory2, dtype: int64
#%% [markdown]
# ## Patients with Positive Pre-op COVID test
#%%
# Number of patienst that have had a positive SARS-CoV-2 PCR test
cases_df.HasPositivePreopCovidTest.value_counts()
# False    34307
# True       996
# Name: HasPositivePreopCovidTest, dtype: int64
#%%
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
sns.set(
    context="notebook",
    style="darkgrid",
    palette="deep",
    font="sans-serif",
    font_scale=1,
    color_codes=True,
    rc={"figure.figsize": (11.7, 8.27)},
)

fig, ax = plt.subplots(figsize=(16, 8))
sns.swarmplot(
    data=cases_with_positive_preop_covid.LastPositiveCovidInterval.dt.days,
    orient="h",
    size=3,
    ax=ax,
)
ax.set(
    xlabel="Number of Days Since Positive SARS-CoV-2 PCR Test",
    title="Time Interval Between PCR-confirmed COVID19 Infection and Surgery",
)
# 0-2 weeks
ax.fill_betweenx(y=[-0.5, 0.5], x1=0, x2=14, alpha=0.25, color=colors.TABLEAU_COLORS["tab:blue"])
# 2-4 weeks
ax.fill_betweenx(y=[-0.5, 0.5], x1=14, x2=28, alpha=0.25, color=colors.TABLEAU_COLORS["tab:orange"])
# 4-6 weeks
ax.fill_betweenx(y=[-0.5, 0.5], x1=28, x2=42, alpha=0.25, color=colors.TABLEAU_COLORS["tab:green"])
# 6-8 weeks
ax.fill_betweenx(y=[-0.5, 0.5], x1=42, x2=56, alpha=0.25, color=colors.TABLEAU_COLORS["tab:red"])
# White: >= 8 weeks

#%% [markdown]
# ## Basic Case Info for Patients with history of COVID+ by PCR test
#%%
# Case Count
fig, ax = plt.subplots()
sns.countplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    order=case_data.covid_case_interval_category2.categories[:-1],
    ax=ax,
)
ax.set(
    title="Number of Cases for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
#%%
# ASA score
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ASA,
    hue_order=["ASA Class 1", "ASA Class 2", "ASA Class 3", "ASA Class 4", "ASA Class 5"],
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="ASA for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ASA,
    hue_order=["ASA Class 1", "ASA Class 2", "ASA Class 3", "ASA Class 4", "ASA Class 5"],
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="ASA for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Emergency Modifier
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.IsEmergency,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Emergency Case Designation for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.IsEmergency,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Emergency Case Designation for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Anatomical Location of Surgery
fig, ax = plt.subplots(figsize=(16, 10))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.SurgeryRegion,
    stat="count",
    multiple="dodge",
    ax=ax,
)
loc, labels = plt.xticks()
ax.set(
    title="Anatomical Region for Surgery/Procedure for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
ax.set_xticklabels(labels, rotation=45)
for x in ax.containers:
    ax.bar_label(x, label_type="edge")

#%%
# Admission Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.AdmissionType,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Admission Type for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.AdmissionType,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Admission Type for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Anesthesia Case Duration
fig, ax = plt.subplots()
sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    y=(cases_with_positive_preop_covid.AnesDuration.dt.total_seconds() / 3600).astype(int),
    order=case_data.covid_case_interval_category2.categories[:-1],
    ax=ax,
)
ax.set(
    title="Anesthesia Case Duration for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Anesthesia Case Duration (hours)",
)
#%%
# PACU Duration
fig, ax = plt.subplots()
sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    y=(cases_with_positive_preop_covid.PACU_Duration.dt.total_seconds() / 60).astype(int),
    order=case_data.covid_case_interval_category2.categories[:-1],
    ax=ax,
)
ax.set(
    title="PACU Duration for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="PACU Duration (minutes)",
)

#%%
# Length of Stay
fig, ax = plt.subplots()
sns.violinplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    y=cases_with_positive_preop_covid.Postop_LOS.dt.days,
    order=case_data.covid_case_interval_category2.categories[:-1],
    ax=ax,
)
ax.set(
    title="Post-op Length of Stay for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
    ylabel="Post-op Length of Stay (days)",
)
#%% [markdown]
# ## Comorbidities (by Elixhauser & MPOG definitions)
# MPOG dataset says that these are taken from ICD billing codes and cannot confirm
# the time point at which they are present--i.e. we do not know if these are present
# pre-operatively vs. post-operatively and if this is a new diagnosis
#%%
# Elixhauser Cardiac Arrhythmias
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserCardiacArrhythmias,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Cardiac Arrhythmias for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserCardiacArrhythmias,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Cardiac Arrhythmias for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Chronic Pulmonary Disease
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserChronicPulmonaryDisease,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Chronic Pulmonary Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserChronicPulmonaryDisease,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Chronic Pulmonary Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Congestive Heart Failure
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserCongestiveHeartFailure,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Congestive Heart Failure for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserCongestiveHeartFailure,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Congestive Heart Failure for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Diabetes Complicated
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserDiabetesComplicated,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Diabetes Complicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserDiabetesComplicated,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Diabetes Complicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Diabetes Uncomplicated
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserDiabetesUncomplicated,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Diabetes Uncomplicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserDiabetesUncomplicated,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Diabetes Uncomplicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Hypertension Complicated
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserHypertensionComplicated,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Hypertension Complicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserHypertensionComplicated,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Hypertension Complicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Hypertension Uncomplicated
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserHypertensionUncomplicated,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Hypertension Uncomplicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserHypertensionUncomplicated,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Hypertension Uncomplicated for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Liver Disease
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserLiverDisease,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Liver Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserLiverDisease,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Liver Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Metastatic Cancer
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserMetastaticCancer,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Metastatic Cancer for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserMetastaticCancer,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Metastatic Cancer for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Obesity
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserObesity,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Obesity for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserObesity,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Obesity for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Peripheral Vascular Disorders
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserPeripheralVascularDisorders,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Peripheral Vascular Disorders for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserPeripheralVascularDisorders,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Peripheral Vascular Disorders for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Pulmonary Circulation Disorders
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserPulmonaryCirculationDisorders,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Pulmonary Circulation Disorders for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserPulmonaryCirculationDisorders,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Pulmonary Circulation Disorders for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Renal Failure
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserRenalFailure,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Renal Failure for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserRenalFailure,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Renal Failure for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Elixhauser Valvular Disease
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserValvularDisease,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Elixhauser Valvular Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityElixhauserValvularDisease,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Elixhauser Valvular Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# MPOG Cerebrovascular Disease
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityMpogCerebrovascularDisease,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="MPOG Cerebrovascular Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityMpogCerebrovascularDisease,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="MPOG Cerebrovascular Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# MPOG Coronary Artery Disease
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityMpogCoronaryArteryDisease,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="MPOG Coronary Artery Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.ComorbidityMpogCoronaryArteryDisease,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="MPOG Coronary Artery Disease for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%% [markdown]
# ## Post-op Complications (by MPOG and AHRQ definitions)
# Presumed that this records only new post-op complications, but MPOG phenotypes do not
# clearly define how all of these are computed
#%%
# Post-op Pulmonary Complications
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadPulmonaryComplication,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadPulmonaryComplication,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Post-op Pulmonary Complication (AHRQ + MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Post-op Cardiac Complications
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadCardiacComplication,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Post-op Cardiac Complication (AHRQ + MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadCardiacComplication,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Post-op Cardiac Complication (AHRQ + MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Post-op Myocardial Infarction Complications
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadMyocardialInfarctionComplication,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Post-op Myocardial Infarction Complication (AHRQ Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadMyocardialInfarctionComplication,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Post-op Myocardial Infarction Complication (AHRQ Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# Post-op AKI Complications
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadAKIComplication,
    stat="percent",
    multiple="fill",
    ax=ax[0],
)
ax[0].set(
    title="Post-op AKI Complication (MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[0].containers:
    ax[0].bar_label(x, label_type="center", fmt="%.2f")

sns.histplot(
    x=cases_with_positive_preop_covid.LabCaseIntervalCategory2,
    hue=cases_with_positive_preop_covid.HadAKIComplication,
    stat="count",
    multiple="dodge",
    ax=ax[1],
)
ax[1].set(
    title="Post-op AKI Complication (MPOG Definition) for Pre-op COVID Category",
    xlabel="Last Positive SARS-CoV-2 PCR+ prior to Procedure",
)
for x in ax[1].containers:
    ax[1].bar_label(x, label_type="edge", fmt="%g")
#%%
# TODO:
# One more possible stratification:
# Get data on whether patient has date of COVID vaccinations (full vs booster)
#   - bool: vaccinated or not?
#   - time since last vaccination or booster?
