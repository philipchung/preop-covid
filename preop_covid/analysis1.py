#%%
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from tqdm.auto import tqdm
from utils import clean_covid_result_value, create_uuid

# Load Data
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
cohort_details_path = data_dir / "raw/v1/Cohort Details_9602.csv"
diagnosis_path = data_dir / "raw/v1/Results_DiagnosesCleanedAggregated_9602.csv"
labs_path = data_dir / "raw/v1/Results_Laboratories_9602.csv"
cases_path = data_dir / "raw/v1/Results_Main_Case_9602.csv"
summary_path = data_dir / "raw/v1/Summary_9602.csv"

raw_cohort_df = pd.read_csv(cohort_details_path)
raw_diagnosis_df = pd.read_csv(diagnosis_path)
raw_labs_df = pd.read_csv(labs_path)
raw_cases_df = pd.read_csv(cases_path)
raw_summary_df = pd.read_csv(summary_path)

# %%
covid_lab_result = CategoricalDtype(categories=["Negative", "Positive", "Unknown"], ordered=False)
covid_case_interval_category = CategoricalDtype(
    categories=["0-2_weeks", "3-4_weeks", "5-6_weeks", ">=7_weeks"], ordered=True
)


def format_labs_df(labs_df: pd.DataFrame) -> pd.DataFrame:
    """Clean labs dataframe.  This returns a new dataframe with different column values.
    Generates a UUID for each lab value, which is unique as long as we have unique
    MPOG Patient ID, Lab Concept ID, and Lab Observation DateTime.  Then we uniquify lab
    values based on this UUID and use it as an index.

    Args:
        labs_df (pd.DataFrame): raw labs dataframe

    Returns:
        pd.DataFrame: transformed output dataframe
    """
    # Create UUID for each Lab Value based on MPOG Patient ID, Lab Concept ID, Lab DateTime.
    # If there are any duplicated UUIDS, this means the row entry has the same value for all 3 of these.
    _df = copy.deepcopy(labs_df)
    lab_uuid = _df.apply(
        lambda row: create_uuid(
            str(row.MPOG_Patient_ID)
            + str(row.MPOG_Lab_Concept_ID)
            + str(row.AIMS_Lab_Observation_DT)
        ),
        axis=1,
    )
    # Clean COVID result column
    lab_result = _df.AIMS_Value_Text.apply(clean_covid_result_value).astype(covid_lab_result)
    # Format String date into DateTime object
    lab_datetime = _df.AIMS_Lab_Observation_DT.apply(
        lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
    )
    # Notes:
    # - Drop "AIMS_Value_Numeric" and "AIMS_Value_CD" columns because they have no
    # information and are all `nan` values.
    # - Drop "MPOG_Lab_Concept_ID" and "Lab_Concept_Name" because this table is only
    # - Lab_Concept_Name='Virology - Coronavirus (SARS-CoV-2)', MPOG_Lab_Concept_ID=5179\
    output_df = (
        pd.DataFrame(
            {
                "LabUUID": lab_uuid,
                "MPOG_Patient_ID": _df.MPOG_Patient_ID,
                "Result": lab_result,
                "DateTime": lab_datetime,
            }
        )
        .drop_duplicates(subset="LabUUID")
        .sort_values(by=["MPOG_Patient_ID", "DateTime"], ascending=[True, True])
        .set_index("LabUUID")
    )
    return output_df


labs_df = format_labs_df(raw_labs_df)


def format_cases_df(cases_df: pd.DataFrame) -> pd.DataFrame:
    """Clean cases dataframe.  This returns a new dataframe with different column values.
    Sets MPOG_Case_ID as index.

    Cases with missing values for Postop_LOS are set to 0 (assume no length of stay).

    Args:
        cases_df (pd.DataFrame): raw cases dataframe

    Returns:
        pd.DataFrame: transformed output dataframe
    """
    _df = copy.deepcopy(cases_df)

    # Format Anesthesia Start into DateTime object
    anes_start = _df.AnesthesiaStart_Value.apply(
        lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
    )
    # Format Case Duration (min) into TimeDelta object
    anes_duration = _df.AnesthesiaDuration_Value.apply(lambda minutes: timedelta(minutes=minutes))
    # Format PACU Duration into TimeDelta object
    pacu_duration = _df.PacuDuration_Value.apply(lambda minutes: timedelta(minutes=minutes))
    # Format Post-op Length of Stay into TimeDelta object
    postop_los_duration = _df.PostopLengthOfStayDays_Value.fillna(0).apply(
        lambda days: timedelta(days=days)
    )

    # Get all ICD codes for pulmonary complications
    pulm_ahrq_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_AHRQ_Diagnoses.replace(
        np.nan, None
    ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
    pulm_mpog_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_MPOG_Diagnoses.replace(
        np.nan, None
    ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
    pulm_icd_codes = (pulm_ahrq_diagnoses + pulm_mpog_diagnoses).apply(set).apply(list)

    output_df = pd.DataFrame(
        {
            "MPOG_Case_ID": _df.MPOG_Case_ID,
            "MPOG_Patient_ID": _df.MPOG_Patient_ID,
            "AnesStart": anes_start,
            "Duration": anes_duration,
            "Age": _df.AgeInYears_Value,
            "Race": _df.Race_Value,
            "Sex": _df.Sex_Value,
            "ASA": _df.AsaStatusClassification_Value,
            "BMI": _df.BodyMassIndex_Value,
            "Weight": _df.Weight_Value,
            "Height": _df.Height_Value,
            "Mortality30Day": _df.HospitalMortality30Day_Value,
            "MORT01": _df.MORT01_Result_Reason,
            "PACU_Duration": pacu_duration,
            "Postop_LOS": postop_los_duration,
            "SurgeryRegion": _df.PrimaryAnesthesiaCPT_MPOGAnesCPTClass,
            "CardiacProcedureType": _df.ProcedureTypeCardiacAlt_Value,
            "IsCardiacProcedure": _df.ProcedureTypeCardiacAlt_Value == "No",
            "PulmonaryComplication": _df.ComplicationAHRQPulmonaryAll_Value,
            "PulmonaryComplicationICD": pulm_icd_codes,
        }
    ).set_index("MPOG_Case_ID")
    return output_df


cases_df = format_cases_df(raw_cases_df)
#%%


def categorize_duration(duration: timedelta) -> str | None:
    """Creates buckets of duration similar to how COVIDSurg 2021 paper did it.
    'Timing of surgery following SARS-CoV-2 infection: an international prospective cohort study'
    (https://pubmed.ncbi.nlm.nih.gov/33690889/)

    Args:
        duration (timedelta): duration of time used to create category/bin.

    Returns:
        str: String description of category of covid case interval.
            For time durations of 0 and negative, None is returned.
    """
    if duration <= timedelta(seconds=0):
        return None
    elif duration < timedelta(weeks=3):
        return "0-2_weeks"
    elif duration >= timedelta(weeks=3) and duration < timedelta(weeks=5):
        return "3-4_weeks"
    elif duration >= timedelta(weeks=5) and duration < timedelta(weeks=7):
        return "5-6_weeks"
    else:
        return ">=7_weeks"


def last_covid_lab_before_case(
    labs_df: pd.DataFrame, mpog_patient_id: str, case_start: pd.Timestamp | datetime
) -> dict[str, Any]:
    """Get most recent COVID lab test before case start datetime.  Lab values occuring
    after case_start are ignored.

    Args:
        labs_df (pd.DataFrame): cleaned labs dataframe
        mpog_patient_id (str): MPOG unique patient identifier
        case_start (pd.Timestamp | datetime): anesthesia start time

    Returns:
        dict[str, Any]: dictionary of most recent covid result as well as whether patient
            had positive covid result lab test in the stratified time intervals:
            0-2 weeks, 3-4 weeks, 5-6 weeks, >7 weeks.
    """
    labs = labs_df.loc[labs_df.MPOG_Patient_ID == mpog_patient_id, :].sort_values(
        by="DateTime", ascending=True
    )
    # Create Timedelta objects for duration interval between lab and case
    labs["LabCaseInterval"] = case_start - labs.DateTime
    labs["LabCaseIntervalCategory"] = (
        labs["LabCaseInterval"].apply(categorize_duration).astype(covid_case_interval_category)
    )
    # Get Only Pre-op Labs Prior to Case
    labs["PreopLab"] = labs.DateTime < case_start
    preop_labs = labs.loc[labs.PreopLab, :]
    # Get Only Positive Pre-op Labs Prior to Case
    pos_preop_labs = preop_labs.loc[preop_labs.Result == "Positive"]
    has_positive_preop_covid_test = not pos_preop_labs.empty
    if has_positive_preop_covid_test:
        last_positive_preop_covid_test = pos_preop_labs.tail(n=1)
        last_positive_preop_covid_labuuid = last_positive_preop_covid_test.index.item()
        last_positive_preop_covid_datetime = last_positive_preop_covid_test.DateTime.item()
        last_positive_preop_covid_interval = last_positive_preop_covid_test.LabCaseInterval.item()
        last_positive_preop_covid_interval_category = (
            last_positive_preop_covid_test.LabCaseIntervalCategory.item()
        )
    else:
        last_positive_preop_covid_labuuid = None
        last_positive_preop_covid_datetime = None
        last_positive_preop_covid_interval = None
        last_positive_preop_covid_interval_category = None

    return {
        "EverCovidPositive": any(preop_labs.Result == "Positive"),
        "HasPositivePreopCovidTest": has_positive_preop_covid_test,
        "LastPositivePreopCovidLabUUID": last_positive_preop_covid_labuuid,
        "LastPositivePreopCovidDateTime": last_positive_preop_covid_datetime,
        "LastPositivePreopCovidInterval": last_positive_preop_covid_interval,
        "LastPositivePreopCovidIntervalCategory": last_positive_preop_covid_interval_category,
    }


#%%
# Jointly iterate through cases and labs by patient ID and associate preop labs with cases
# This takes ~5 min to run.
labs_for_cases_for_all_patients = []
for cases_grp, labs_grp in tqdm(
    zip(cases_df.groupby("MPOG_Patient_ID"), labs_df.groupby("MPOG_Patient_ID")),
    desc="Associating Labs to Cases",
):
    mpog_case_id, cases = cases_grp
    lab_uuid, labs = labs_grp

    labs_for_cases = cases.apply(
        lambda row: last_covid_lab_before_case(
            labs_df=labs, mpog_patient_id=row.MPOG_Patient_ID, case_start=row.AnesStart
        ),
        axis=1,
    ).apply(pd.Series)
    labs_for_cases_for_all_patients += [labs_for_cases]

labs_for_cases_for_all_patients = pd.concat(labs_for_cases_for_all_patients)

#%%
# Combine extracted case-level lab info into the Cases Dataframe
cases_df2 = cases_df.join(labs_for_cases_for_all_patients)
#%%
# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df2.loc[cases_df2.HasPositivePreopCovidTest]
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
    y=[-0.4, 0.4],
    x1=0,
    x2=17,
    alpha=0.5,
    color=matplotlib.colors.TABLEAU_COLORS["tab:orange"]
)
# Red = 3-4 weeks (actually 2.5-4.5 weeks)
p1.fill_betweenx(
    y=[-0.4, 0.4],
    x1=17,
    x2=31,
    alpha=0.5,
    color=matplotlib.colors.TABLEAU_COLORS["tab:red"]
)
# Cyan = 5-6 weeks (actually 4.5-7.5 weeks)
p1.fill_betweenx(
    y=[-0.4, 0.4],
    x1=31,
    x2=45,
    alpha=0.5,
    color=matplotlib.colors.TABLEAU_COLORS["tab:cyan"]
)
# White: >= 7 weeks (actually 7.5+ weeks)


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
