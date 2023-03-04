#%%
import re
from pathlib import Path

import pandas as pd
import yaml
from case_data import CaseData
from lab_data import LabData
from pandas.api.types import CategoricalDtype

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

#%%
# Load ROS Table
ros_df = pd.read_csv(ros_path)
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


#%%
# Load Health Maintenance / Vaccine Table
hm_df = pd.read_csv(hm_path)
# Drop Incompleted Vaccines
vaccines_df = hm_df.loc[hm_df.HMType == "Done"]
# Clean Vaccine Type
vaccines_df["VaccineType"] = vaccines_df.HMTopic.apply(clean_hm_topic_value).astype(
    vaccine_category
)
# Keep Only COVID-19 and Influenza Vaccines
vaccines_df = vaccines_df.loc[vaccines_df.VaccineType != "Other"]
vaccines_df
#%%
# TODO:
# - parse comments -- need to do separately for COVID-19 & Flu
vaccines_covid = vaccines_df.loc[vaccines_df.VaccineType == "COVID-19"]
vaccines_flu = vaccines_df.loc[vaccines_df.VaccineType == "Influenza"]
#%% [markdown]
# ## Clean up COVID Vaccine Metadata
#%%
pattern = (
    "(?P<AdminCategory>.*):\s(?P<LIMID>LIMI[dD].*?)?\^?(?P<LPLID>LPLId.*?)?\^\\x11(?P<Vaccine>.*)"
)

res = vaccines_covid.COMMENTS.apply(lambda x: re.search(pattern, x))
grp_names = ["AdminCategory", "LIMID", "LPLID", "Vaccine"]
print({x: res.iloc[0].group(x) for x in grp_names})
print(res.iloc[0].groups())

res1 = res.apply(lambda match: {name: match.group(name) for name in grp_names})

res2 = pd.DataFrame(res1.tolist())
res2.Vaccine = res2.Vaccine.str.rstrip("\x11\x110").str.strip()


#%%
# Vaccine Crosswalk Codes taken from CDC
# https://www.cdc.gov/vaccines/programs/iis/COVID-19-related-codes.html
covid_vaccine_codes = pd.read_csv(data_dir / "v2" / "mappings" / "vaccine_codes.csv")

moderna_cvx_short_description = covid_vaccine_codes.loc[
    covid_vaccine_codes.Manufacturer == "Moderna US, Inc."
]["CVX Short Description"].tolist()

pfizer_cvx_short_description = covid_vaccine_codes.loc[
    covid_vaccine_codes.Manufacturer == "Pfizer-BioNTech"
]["CVX Short Description"].tolist()

janssen_cvx_short_description = covid_vaccine_codes.loc[
    covid_vaccine_codes.Manufacturer == "Janssen Products, LP"
]["CVX Short Description"].tolist()

novavax_cvx_short_description = covid_vaccine_codes.loc[
    covid_vaccine_codes.Manufacturer == "Novavax, Inc."
]["CVX Short Description"].tolist()

sanofi_cvx_short_description = covid_vaccine_codes.loc[
    covid_vaccine_codes.Manufacturer == "Sanofi Pasteur"
]["CVX Short Description"].tolist()


covid_vaccine_kind = CategoricalDtype(
    categories=[
        "Moderna",
        "Pfizer",
        "Janssen",
        "Novavax",
        "Sanofi",
        "AstraZeneca",
        "Unspecified mRNA Vaccine",
        "Other",
    ],
    ordered=False,
)


def categorize_covid_vaccine_kind(text: str) -> str:
    text_lower = text.lower()

    # TODO: match covid_vaccine_codes fields
    # CVX Short Description & CVX Long Description & Sale Proprietary Names
    # to the vaccine text descriptions
    # - may also need to strip out \x11\x110 tail
    if text in moderna_cvx_short_description:
        return "Moderna"
    if "moderna" in text_lower:
        return "Moderna"
    elif "tozinameran" in text_lower:
        return "Moderna"
    elif "mRNA-1273" in text:
        return "Moderna"
    elif text in pfizer_cvx_short_description:
        return "Pfizer"
    elif "pfizer" in text_lower:
        return "Pfizer"
    elif "BNT-162b2" in text:
        return "Pfizer"
    elif text in janssen_cvx_short_description:
        return "Janssen"
    elif "janssen" in text_lower:
        return "Janssen"
    elif "johnson" in text_lower:
        return "Janssen"
    elif "rS-Ad26" in text:
        return "Janssen"
    elif "Ad26" in text:
        return "Janssen"
    elif text in novavax_cvx_short_description:
        return "Novavax"
    elif "novavax" in text_lower:
        return "Novavax"
    elif "rS-nanoparticle+Matrix-M1" in text:
        return "Novavax"
    elif text in sanofi_cvx_short_description:
        return "Sanofi"
    elif "sanofi" in text_lower:
        return "Sanofi"
    elif "astrazeneca" in text_lower:
        return "AstraZeneca"
    elif "rS-ChAdOx1" in text:
        return "AstraZeneca"
    elif "mRNA" in text:
        return "Unspecified mRNA Vaccine"
    else:
        return "Other"


##
res3 = copy.deepcopy(res2)

res3["covid_vaccine_kind"] = res3.Vaccine.apply(categorize_covid_vaccine_kind).astype(
    covid_vaccine_kind
)
res3.covid_vaccine_kind.value_counts()

#%%
res3.loc[res3.covid_vaccine_kind == "Other"].Vaccine.unique().tolist()

#%%
