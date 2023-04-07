#%%
from pathlib import Path

import pandas as pd
from case_data import CaseData
from lab_data import LabData
from preop_data import PreopSDE
from vaccine_data import VaccineData

from .utils.parallel_process import parallel_process

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
# Associate Labs from each Patient to Cases for each Patient
cases_df = case_data.associate_labs_to_cases(labs_df=labs_df)

# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

#%%
# Load & Clean Vaccine Data
vaccine_data = VaccineData(vaccines_df=hm_path, data_version=data_version)
flu_vaccines_df = vaccine_data.flu_vaccines_df
covid_vaccines_df = vaccine_data.covid_vaccines_df

covid_vaccines_df.VaccineKind.value_counts()
#%%
# Load & Clean SmartDataElements Data
preop_data = PreopSDE(preop_df=preop_smartdataelements_path, data_version=data_version)
problems_df = preop_data.problems_df

problems_df.loc[problems_df.IsPresent].Problem.value_counts()

#%%
# TODO:
# 1. pick out top diagnoses (CV, pulm, renal) from ROS/problems_df that we want to examine
# 2. join ROS/problems into cases_df table (alongside Elixhauser Comorbidities)
#    so each case has associated ROS (binary values)
# 3. for each case, get total # of vaccines/boosters; get # of vaccines/booster & duration
#    since last vaccine/booster administration.
# 4. For patients with vaccine in past 6 months, compare outcomes (PACU duration, Hospital LOS, Mortality)
# 5. repeat for patients with vaccine in past 12 months...
# (NEJM paper suggests clinical protection for 6 months: https://www.nejm.org/doi/full/10.1056/NEJMoa2118691)
# 6. Patients with Heart Disease

#%%

# %%
c = cases_df.loc[:, ["MPOG_Patient_ID", "AnesStart"]]

# Join Covid Vaccines to Cases
# (row for every unique MPOG_Case_ID & Vaccine_UUID combination)
df = pd.merge(
    left=c.reset_index(),
    right=covid_vaccines_df.reset_index(),
    how="inner",
    left_on="MPOG_Patient_ID",
    right_on="MPOG_Patient_ID",
)
df
#%%
# Only keep rows where vaccine was administered prior to Case Start
res = df.loc[df.AnesStart > df.VaccineDate]
res
#%%
from datetime import timedelta

from utils import parallel_process

# Compute Durations Between Vaccine
gb = res.groupby("MPOG_Case_ID")[["AnesStart", "VaccineUUID", "VaccineDate", "VaccineKind"]]


def vaccine_durations_per_case(MPOG_Case_ID: str, df: pd.DataFrame) -> dict:
    d = vaccine_durations(df)
    return {"MPOG_Case_ID": MPOG_Case_ID} | d


def vaccine_durations(df: pd.DataFrame) -> dict:
    """Compute vaccine durations between case and vaccine administraiton.

    Args:
        df (pd.DataFrame): Dataframe with columns
            ["AnesStart", "VaccineUUID", "VaccineDate", "VaccineKind"]

    Returns:
        dict: dictionary of derived duration values & categories.
    """
    _df = df.copy()
    # Only keep rows where vaccine was administered prior to Case Start
    _df = _df.loc[_df.AnesStart > _df.VaccineDate]
    # Durations
    duration = _df.AnesStart - _df.VaccineDate

    return {
        "NumCovidVaccines": len(_df),
        "CovidVaccineDateTimes": _df.VaccineDate.tolist(),
        "CovidVaccineCaseIntervals": duration.tolist(),
        "CovidVaccineLastThreeMonths": any(duration < timedelta(days=90)),
        "CovidVaccineLastSixMonths": any(duration < timedelta(days=180)),
        "CovidVaccineLastTwelveMonths": any(duration < timedelta(days=365)),
    }


results = parallel_process(
    iterable=gb, function=vaccine_durations_per_case, use_args=True, desc="Vaccine Case Durations"
)
#%%
df2 = pd.DataFrame(results)

#%%
# %%
# limit to possible covid?  home test?
# look just at vaccination status.  Difference in all-comers.

# - is there a difference risk between 2-3 post-op complications between vaccinated & non-vaccinated

# ROS groups:
# - COPD (assoc. w/ COVID)
# - CHF (interaction w/ COVID)
# - h/o MI
# - stroke (inc. risk w/ stroke w/ COVID)
