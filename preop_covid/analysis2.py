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
#%%
# pattern = "(.*):(.*)^(.*)^\\x11(.*)"
# xxx: LIMID-xxx ^ (optional LPLId-xxxxxxxx) ^\x11 VaccineType
pattern = "(.*):\s(LIMI[dD].*)?\^?(LPLId.*)?\^\\x11(.*)(?:\\x11?)(?:\\x110?)"

# text = vaccines_covid.COMMENTS.head(20).tolist()[0]
text = "Imm Admin: LIMId-153^LPLId-19924636^\x11COVID-19 Janssen vector-nr rS-Ad26"
print(text)

match = re.search(pattern, text)
match.groups()
#%%
pattern = (
    "(?P<AdminCategory>.*):\s(?P<LIMID>LIMI[dD].*?)?\^?(?P<LPLID>LPLId.*?)?\^\\x11(?P<Vaccine>.*)"
)

res = vaccines_covid.COMMENTS.apply(lambda x: re.search(pattern, x))
grp_names = ["AdminCategory", "LIMID", "LPLID", "Vaccine"]
print({x: res.iloc[0].group(x) for x in grp_names})
res.iloc[0].groups()
#%%
res1 = res.apply(lambda match: {name: match.group(name) for name in grp_names})

res2 = pd.DataFrame(res1.tolist())
#%%
res2.Vaccine.value_counts()
# COVID-19 Pfizer mRNA 12 yrs and older (purple cap)                                39090
# COVID-19 Moderna mRNA 12 yrs and older                                            34728
# COVID-19 Pfizer mRNA purple cap                                                   28249
# COVID-19 Moderna mRNA LNP-S                                                       19455
# COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose0                                  12818
#                                                                                   ...
# SARS-COV-2 (COVID-19)Booster0                                                       1
# Covid Vaccine Pfizer-Biontech0                                                      1
# MODERNA COVID-19 VACCINE (GREY LABEL/BLUE CAP), 12+ BIVALENT BOOSTER0               1
# SARS-CoV-2 (COVID-19) mRNA BNT-162b2 vax20                                          1
# SARS-CoV-2 (COVID-19) vaccine (Pfizer), mRNA, LNP-S, PF, 30 mcg/0.3 mL dose0        1
# Name: Vaccine, Length: 434, dtype: int64

# NOTE: there are 434 possible values, probably because these are administered across the state
# and each site has different way of recording it

#%%
covid_vaccine_codes = pd.read_csv(data_dir / "v2" / "mappings" / "vaccine_codes.csv")
covid_vaccine_codes


def categorize_covid_vaccine(text: str) -> str:
    s = text.lower()

    # TODO: match covid_vaccine_codes fields
    # CVX Short Description & CVX Long Description & Sale Proprietary Names
    # to the vaccine text descriptions
    # - may also need to strip out \x11\x110 tail

    if "moderna" in s:
        return "Moderna"
    elif "tozinameran" in s:
        return "Moderna"
    elif "pfizer" in s:
        return "Pfizer"
    elif "janssen" in s:
        return "Janssen"
    elif "novavax" in s:
        return "Novavax"
    elif "sanofi" in s:
        return "Sanofi"


#%%
res2.Vaccine.unique().tolist()
# ['COVID-19 Janssen vector-nr rS-Ad26',
#  'COVID-19 Moderna mRNA 12 yrs and older',
#  'COVID-19 Pfizer mRNA tris-sucrose 12 yrs and older (gray cap)',
#  'COVID-19 Pfizer mRNA 12 yrs and older (purple cap)',
#  'COVID-19 Pfizer mRNA bivalent booster 12 yrs and older',
#  'SARS-CoV-2 mRNA (tozinameran) vaccine\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 10 mcg/0.2 mL dose, tris-sucrose\x11\x110',
#  'COVID-19 Pfizer mRNA purple cap',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/ 0.5 mL dose\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5 mL dose\x11\x110',
#  'Moderna COVID-19 Vaccine, red cap blue label, 12+ Primary Series\x11\x110',
#  'COVID-19 (MODERNA, LIGHT BLUE LABEL-BORDER) 12 YRS+, 0.5 ML\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, mRNA, spike protein, LNP, preservative free, 100 mcg/0.5mL dose\x11\x110',
#  'COVID-19 (MODERNA) BIVALENT BOOSTER 12 YRS+ 0.5 ML\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'COVID-19 Pfizer mRNA tris-sucrose 5-11 years old',
#  'COVID-19 Moderna 50 mcg/0.5 mL Booster dose: BIVALENT VACCINE (Blue capped vial with gray-bordered label) 12+yrs\x11\x110',
#  'COVID-19 vaccine, vector-nr, rS-Ad26, PF, 0.5 mL\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5mL dose or 50 mcg/0.25mL dose\x11\x110',
#  'COVID-19 Moderna mRNA LNP-S',
#  'COVID-19 Moderna mRNA bivalent booster 6 yrs and older',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose, tris-sucrose\x11\x110',
#  'COVID-19 mRNA LMP-S PF 30 Mcg/0.3 Ml (Pfizer) 12+yrs\x11\x110',
#  'Moderna COVID-19 Vaccine Bivalent Booster, blue cap, 6+\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'COVID-19 Pfizer mRNA tris-sucrose gray cap',
#  'COVID-19 Pfizer mRNA tris-sucrose 5-11 yrs old',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg or 50 mcg dose\x11\x110',
#  'COVID-19 Moderna mRNA bivalent booster 18 yrs and older',
#  'Pfizer Sars-cov-2 Vaccination 12+ y.o.(Purple)\x11\x110',
#  'COVID-19 MRNA (PFIZER) I 30 MCG/0.3 ML\x11\x110',
#  'COVID-19 MRNA (MODERNA) 50 MCG/0.25 ML\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 50 mcg/0.5 mL or 25mcg/0.25 mL dose\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA-1273 vaccine\x11\x110',
#  'Janssen COVID-19 Vaccine\x11\x110',
#  'COVID-19 Moderna 100 mcg/0.5 mL Primary series: MONOVALENT VACCINE (Red capped vial with blue-bordered label) 12+yrs\x11\x110',
#  'COVID-19 (PFIZER, PURPLE CAP) 12 YRS+, 0.3 mL (ORIGINAL)\x11\x110',
#  'MODERNA COVID BIVAL(18Y UP)EUA\x11\x110',
#  'Moderna SARS-CoV-2 Vaccination (12 + y/o)(RED CAP)\x11\x110',
#  'SARS-COV-2 (COVID-19) VACCINE PRIMARY SERIES (PFIZER), MRNA, LNP-S, PF, 30 MCG/0.3 ML DOSE\x11\x110',
#  'COVID-19 Moderna mRNA 6-11 yrs old or adult booster (50 mcg/0.5 mL)',
#  'Pfizer SARS-CoV-2 Vaccination (Purple Cap)\x11\x110',
#  'COVID-19 Pfizer mRNA LNP-S',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ2, 3\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ1\x11\x110',
#  'COVID-19 AstraZeneca vector-nr rS-ChAdOx1',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5 mL dose (Moderna)\x11\x110',
#  'COVID-19 Moderna mRNA 18 yrs and older',
#  'COVID-19 VACCINE, VECTOR-NR, RS-AD26 (JANSSEN, J&J) PF 0.5 ML\x11\x110',
#  'COVID-19 Moderna mRNA bivalent booster 6 mos-5 yrs old',
#  'COVID-19 (UNSPECIFIED)\x11\x110',
#  'COVID-19 Pfizer mRNA tris-sucrose 6 mos-4 yrs old',
#  'Moderna SARS-CoV-2 Bivalent Vaccination\x11\x110',
#  'COVID-19 (SARS-COV-2) vaccine, unspecified\x11\x110',
#  'Pfizer SARS-CoV-2 Vaccination (12 + y/o) (PURPLE CAP)\x11\x110',
#  'COVID-19 Pfizer mRNA bivalent booster 5-11 yrs old',
#  'Pfizer COVID Vaccine, purple cap, 12+\x11\x110',
#  'Moderna 18yr and up bivalent booster\x11\x110',
#  'COVID19 Moderna Bivalent Boost 6 or 12y+\x11\x110',
#  'COVID-19, unspecified',
#  'COVID-19 Moderna mRNA 6 mos-5 yrs old',
#  'COVID-19 (Pfizer)\x11\x110',
#  'Moderna Sars-cov-2 Vaccination\x11\x110',
#  'COVID-19 vaccine, vector-nr, rS-Ad26, PF, 0.5 mL (Janssen)\x11\x110',
#  'Pfizer\x111\x110',
#  'COVID-19 Pfizer BioNTech\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, mRNA, spike protein, LNP, preservative free, 30 mcg/0.3mL dose\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster (Pfizer), PF, 30 mcg/0.3 mL dose Dosage\x11\x110',
#  'COVID-19 Pfizer 30 mcg/0.3 mL Primary series: MONOVALENT VACCINE (Gray capped vial with gray-bordered label) 12+yrs\x11\x110',
#  'COVID-19 mRNA LNP-S PF 100 mcg/0.5 mL MODERNA 12+yrs (TPC)\x11\x110',
#  'COVID-19 (PFIZER) BIVALENT BOOSTER 12 YRS+, 0.3 ML\x11\x110',
#  'COVID-19 vaccine, Subunit, rS-nanoparticle+Matrix-M1 Adjuvant, PF, 0.5 mL\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose (Pfizer-BioNTech)\x11\x110',
#  'Pfizer SARS-CoV-2 (COMIRNATY) Vaccination (12 + y/o)(GREY CAP)\x11\x110',
#  'COVID-19 Pfizer bivalent (12yr+ booster), PF - GRAY CAP\x11\x110',
#  'SARS-CoV-2 mRNA BNT-162b2 vaccine\x11\x110',
#  'SARS-COV-2 (COVID-19) vax, unspecified2\x11\x110',
#  'COVID-19 Novavax subunit adjuvanted',
#  'Moderna COVID-19 Vaccine\x11\x110',
#  'COVID-19 MRNA (MODERNA) 100 MCG/0.5 ML\x11\x110',
#  'Pfizer Sars-cov-2 Vaccination\x11\x110',
#  'COVID-19 mRNA LNP-S PF 100 mcg/0.5 mL dose (Moderna)\x11\x110',
#  'Covid-19 Vaccine MRNA (PF) 12yr+ (Pfizer/BioNTech)(IMM601)\x11\x110',
#  'SARS-CoV-2 mRNA tozinameran-tris-sucrose\x11\x110',
#  'Covid-19 (Pfizer 12 yrs+ Purple - Primary) mRNA, LNP-S, PF, 30 mcg/0.3 mL IM (CVX-208)\x11\x110',
#  'SARS-CoV-2(C-19) Ad26-JSN\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+Pri/Bstr)-MOD\x11\x110',
#  'COVID-19 mRNA LNP-S PF 30 mcg/0.3 mL Bivalent Booster (Pfizer) 12+yrs\x11\x110',
#  'COVID-19 (JANSSEN)\x11\x110',
#  'COVID-19 PF Vaccine 100mcg/0.5mL (Moderna)\x11\x110',
#  'COVID-19 Novavax subunit rS-nanoparticle',
#  'Investigational study COVID-19 Moderna mRNA LNP-S',
#  'Pfizer Bivalent Booster 12 and older\x11\x110',
#  'PFIZER COVID (12Y UP) VAC-GRAY\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA BNT-162b2 vax\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+Pri/Bstr)-MOD1\x11\x110',
#  'SARS-CoV-2 (COVID-19)mRNA BNT-162b2 vacc\x11\x110',
#  'COVID-19 Mrna Lnp-s PF 30 Mcg/0.3 Ml (Pfizer) 12+yrs\x11\x110',
#  'SARS-CoV-2 for 12+ YO (Pfizer Purple Cap)\x11\x110',
#  'COVID-19 Pfizer 30 mcg/0.3 mL Booster dose: BIVALENT VACCINE (Gray capped vial with gray-bordered label) 12+yrs\x11\x110',
#  'SARS-CoV-2 mRNA(tozinameran) vaccine-Pfz\x11\x110',
#  'Moderna SARS-CoV-2 Vaccination\x11\x110',
#  'Covid-19 Unspecified - Historical\x11\x110',
#  'COVID-19 mRNA LNP-S PF 50 mcg/0.5 mL Bivalent Booster (Moderna) 18+yrs\x11\x110',
#  'Moderna SARS-CoV-2 Vacc Bivalent Booster 12+\x11\x110',
#  'Moderna Sars-cov-2 Vaccination 12+ y.o.\x11\x110',
#  'COVID-19 (PFIZER) BIVALENT BOOSTER 5-11 YRS+, 0.2 ML\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 10 mcg/0.2 mL dose\x11\x110',
#  'COVID-19 (PFIZER, PURPLE CAP) 12 YRS+, 0.3 mL\x11\x110',
#  'Covid-19 Vaccine (Pfizer)\x11\x110',
#  'Moderna SARS-COV-2 (COVID-19) Vaccine, 100mcg/0.5ml dose\x11\x110',
#  'COVID-19 MRNA BIVALENT (MODERNA) 50 MCG/0.5 ML\x11\x110',
#  'SARS-CoV-2 Pfizer\x11\x110',
#  'Janssen Covid-19 Vaccination\x11\x110',
#  'Moderna COVID-19 Bivalent 0.5ml BOOSTER\x11\x110',
#  'Moderna SARS-CoV-2 BIVALENT Vaccination\x11\x110',
#  'Covid-19 Vaccine MRNA (PF) 18yr+ (Moderna)(IMM600)\x11\x110',
#  'COVID-19 Sinovac inactivated, non-US (CoronaVac)',
#  'COVID (Moderna) - historical only\x11\x110',
#  'COVID-19/PFIZER Bivalent Booster\x11\x110',
#  ' Pfizer SARS-CoV-2 Vaccination (Purple Cap)\x11\x110',
#  'Covid-19 Mrna Lnp-s PF 30 Mcg/0.3 Ml Im (Pfizer)\x11\x110',
#  'SARS-COV-2 (MODERNA) FULL DOSE (0.5ML) 12YRS+\x11\x110',
#  'Pfizer SARS-CoV-2 BIVALENT Vaccination\x11\x110',
#  'COVID-19 mRNA LNP-S, PF (MODERNA),100 MCG/0.50 mL, EXTERNAL ADMINISTRATION\x11\x110',
#  'Moderna\x111\x110',
#  'COVID-19 (Pfizer Bivalent) 12+\x11\x110',
#  'Moderna SARS-CoV-2 Vaccine\x11\x110',
#  'COVID-19 (MODERNA, LIGHT BLUE LABEL-BORDER) 18 yrs+, 0.25 ML BOOSTER\x11\x110',
#  'PFIZER Covid-19, mRNA tris vaccine (PF) 30 mcg/0.3 mL IM (PURPLE TOP)\x11\x110',
#  'COVID-19 Moderna mRNA bivalent booster 6 yrs and older (from claim)\x11\x111',
#  'SARS-COV-2 (COVID-19) vaccine, UNSPECIFIED\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA-1273 vaccine3\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA-1273 vaccine2\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA-1273 vaccine1\x11\x110',
#  'PFIZER COVID BIVAL (12Y UP)EUA\x11\x110',
#  'COVID-19 Mrna Lnp-s PF 30 Mcg/0.3 Ml Im (Pfizer) 12+yrs\x11\x110',
#  'COVID-19 (Moderna)\x11\x110',
#  'COVID-19, mRNA, LNP-S, Bivalent Booster, 30 mcg/0.3 Ml (PFIZER) (GRAY CAP) (12 YRS AND OLDER)\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5 mL dose (Moderna) - ML\x11\x110',
#  'COVID-19 Moderna (12yr+), PF - RED CAP\x11\x110',
#  'COVID-19 Pfizer 30mcg/0.3 mL dose\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi5\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi4\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi3\x11\x110',
#  'SARS-CoV-2 mRNA BNT-162b2 vaccine1\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5 mL\x11\x110',
#  'COVID-19 12Y+ Pfizer-BioNtech - Requires Dilution\x11\x110',
#  'pfizer (purple cap) Sars-cov-2 Vaccination\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, pediatric 25 mcg/0.25 mL dose\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 10 mcg/0.2 mL\x11\x110',
#  'COVID-19 mRNA LNP-S PF 50 mcg/0.5 mL Bivalent Booster (Moderna) 12+yrs\x11\x110',
#  'COVID-19 (PFIZER, GRAY CAP) 12 YRS+, 0.3 ML (COMIRNATY)\x11\x110',
#  'SARS-COV-2 (COVID-19) Booster\x11\x110',
#  'SARS-COV-2 for 12+ YO BOOSTER Bivalent (Moderna Grey Label)\x11\x110',
#  'SARS-COV-2 for 12+ YO 0.5 mL (Moderna Red Label)\x11\x110',
#  'COVID-19 (ASTRAZENECA)\x11\x110',
#  'Pfizer-BioNTech COVID-19 Vaccine Dose 1\x11\x110',
#  'Pfizer-BioNTech COVID-19 Vaccine Dose 2\x11\x110',
#  'COVID-19 (Pfizer-BioNTech) mRNA 30 MCG  Vaccine (12+) (PURPLE CAP)\x11\x110',
#  'Pfizer-BioNTech COVID-19 Vaccine Bivalent Booster, (GRAY CAP) (Tris-sucrose formulation), 12+\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent, PF, 50 mcg/0.5 mL dose (Moderna)\x11\x110',
#  'Pfizer COVID-19 Bivalent 0.3ml BOOSTER\x11\x110',
#  'COVID-19 Pfizer mRNA bivalent 6 mos-4 yrs old',
#  'COVID-19 vaccine, vector-nr, rS-Ad26, PF, 0.5\x11\x110',
#  'SARS-COV-2 (COVID-19) BiValent Booster\x11\x110',
#  'CRX Pfizer Covid-19 (Bivalent) 12+\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, mRNA, spike protein, LNP, preservative free, 30 mcg/0.3mL dose (Pfizer)\x11\x110',
#  'COVID19 MODERNA UPDATED BOOSTR 6-11\x11\x110',
#  'COVID-19 mRNA LNP-S PF 50 mcg/0.25 mL Booster (Moderna)\x11\x110',
#  'SARS-COV-2 (COVID-19)\x11\x110',
#  'COVID-19 vaccine, MRNA, Pfizer, 0.3 ML\x11\x110',
#  'SARS-CoV-2 for 12+ YO BOOSTER Bivalent (Pfizer Gray Cap)\x11\x110',
#  'Moderna Booster\x11\x110',
#  'SARS-COV-2 (COVID-19) VACCINE PRIMARY SERIES (MODERNA), MRNA, LNP-S, PF, 100 mcg/ 0.5 mL dose\x11\x110',
#  'COVID-19 Moderna mRNA 50 mcg/0.5 mL booster dose',
#  'COVID-19 mRNA (PF)(LNP-S BIVALENT BOOSTER) 12YR+ (PFIZER)(IMM301)\x11\x110',
#  'COVID-19 vaccine, vector-nr, rS-Ad26, PF (Janssen)\x11\x110',
#  'COVID-19 MRNA BIVALENT (PFIZER) 30 MCG/0.3 ML\x11\x110',
#  'Pfizer Bivalent Covid Booster (12+)\x11\x110',
#  'COVID-19 Pfizer mRNA 12 yrs and older (purple cap) (from claim)\x11\x111',
#  'SARS-COV-2 (COVID-19) vax, unspecified1, 2\x11\x110',
#  'PFIZER COVID BIVAL(12Y UP)-EUA\x11\x110',
#  'SARS-COV-2 (COVID-19) Moderna BOOSTER\x11\x110',
#  'COVID19 Pfizer Bivalent Booster 12+\x11\x110',
#  'COVID-19 (MODERNA) BIVALENT BOOSTER 18 YRS+, 0.5 ML\x11\x110',
#  'SARS-COV-2 for 18+ YO BOOSTER Bivalent (Moderna Gray Label)\x11\x110',
#  'Moderna Covid\x11\x110',
#  'COVID-19 Vaccine (Moderna)\x11\x110',
#  'COVID-19 mRNA LNP-S PF 100 mcg/0.5 mL MODERNA 12+yrs\x11\x110',
#  'COVID-19, mRNA, (Moderna) 100 mcg/0.5 mL\x11\x110',
#  'COVID-19 PFIZER (16+ YRS)\x11\x110',
#  'SARS-COV-2 (COVID-19) Pfizer BOOSTER\x11\x110',
#  'Covid-19 Moderna 100 mcg/0.5 mL dose\x11\x110',
#  'COVID-19 PF Vaccine, Bivalent Booster (Moderna)\x11\x110',
#  'SARS-CoV-2 (COVID-19) Ad26 vaccine\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose, tri\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ grey)-PFZ1\x11\x110',
#  'COVID-19 (PFIZER, GRAY CAP) 12 YRS+, 0.3 ML\x11\x110',
#  '#Moderna COVID-19, low dose booster\x11\x110',
#  'COVID-19 UNSPECIFIED FORMULATION\x11\x110',
#  'Covid-19 (Moderna Mono, 12yrs+ - Red/Blue-Primary) mRNA, LNP-S, PF 100 mcg/0.5mL IM (CVX-207)\x11\x110',
#  'SARS-CoV-2 for 12+ YO (Pfizer Gray Cap)\x11\x110',
#  'COVID-19, Moderna Booster\x11\x110',
#  'Janssen Sars-cov-2 Vaccination\x11\x110',
#  'COVID-19 Pfizer (12yr+), PF - Original Formulation\x11\x110',
#  'COVID-19 mRNA MONOVALENT vaccine PRIMARY SERIES 12 years and above (Moderna) 100 mcg/0.5 mL\x11\x110',
#  'Pfizer COVID-19 vaccine, preservative free, 30 mcg/0.3mL dose (Pfizer)\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi\x11\x110',
#  'COVID-19 PF Vaccine 30mcg/0.3mL (Pfizer)\x11\x110',
#  'Pfizer-BioNTech COVID19 Vaccine, 0.3mL per dose, 2 doses, administered 21 days apart\x11\x110',
#  'Moderna 12+ Monovalent (Red Cap - DO NOT DILUTE) COVID-19 Vaccine INJ\x11\x110',
#  'Pfizer-BioNTech COVID-19 Vaccine\x11\x110',
#  'SARS-CoV-2(C-19) mRNA-1273 vaccine- Mod\x11\x110',
#  'Pfizer-SARS-CoV-2 Vaccine\x11\x110',
#  'Pfizer SARS-CoV-2 Vaccination (12 + y/o)(Bivalent Booster Dose Formulation)(GREY CAP)\x11\x110',
#  'COVID-19 mRNA LNP-S PF 30 mcg/0.3 mL Bivalent Booster (Pfizer) 12+yrs (TPC)\x11\x110',
#  'Moderna Bivalent 18+ yrs\x11\x110',
#  'COVID-19 Pfizer-BioNTech\x11\x110',
#  'COVID-19 VACCINE MRNA (LNP-S) 6YR+ (MODERNA/BIVALENT BOOSTER)(IMM302)\x11\x110',
#  'COVID-19 Moderna mRNA 12 yrs and older (from claim)\x11\x111',
#  'SARS-CoV-2(C-19) mRNA (12y+Pri/Bstr)-MOD2\x11\x110',
#  'Covid Vaccine (Moderna 28-Day Dosing)\x11\x110',
#  'Covid-19, DNA, Spike Protein,(Ad26) Vector, PF 0.5mL IM (Janssen JNJ)\x11\x110',
#  'SARS-COV-2 (COVID-19) vax, unspecified8\x11\x110',
#  'Moderna (Bivalent Booster) 18YR+ Covid-19, mRNA, Pf, 50 mcg/0.5 mL dose\x11\x110',
#  'COVID-19 vaccine (Pfizer), 0.3 mL, Injectable, mRNA (Purple Top)\x11\x110',
#  'Covid-19 Vaccine (Pfizer), 0.3 Ml, Injectable, Mrna (Gray Top)\x11\x110',
#  'COVID-19 vaccine, vector-nr, rS-ChAdOx1, PF, 0.5 mL\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ3\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ1, 2\x11\x110',
#  'COVID-19 Pfizer mRNA tris-sucrose 12 yrs and older (gray cap)\x11\x110',
#  'Moderna Covid Vaccine  50mcg/0.25ml IM\x11\x110',
#  'Pfizer SARS-CoV-2 Vaccination\x11\x110',
#  'Covid-19 MRNA BIVAL (PFIZER)30MCG/0.3ML IM SUSP\x11\x110',
#  'SARS-CoV-2 for 12+ YO (aka Pfizer Purple Cap)\x11\x110',
#  'SARS-COV-2 (COVID-19) VACCINE, UNSPECIFIED (HISTORICAL USE ONLY)\x11\x110',
#  'SARS-COV-2 (COVID-19) vax, unspecified1\x11\x110',
#  'PFIZER-BIONTECH COVID-19 VACCINEBIVALENTBA.4BA.5 INJ PFIZ\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ grey)-PFZ\x11\x110',
#  'Pfizer COVID vaccine, gray cap, 12+\x11\x110',
#  'COVID-19 Moderna, External Administration\x11\x110',
#  'Pfizer bivalent\x11\x110',
#  'COVID-19 mRNA bivalent booster (Moderna 6+)\x11\x110',
#  'Moderna Age 12Y+ 100 mcg/0.5 mL SARS-CoV-2 Vaccination\x11\x110',
#  'Moderna SARS-CoV-2 Vaccination (12 + y/o)(Bivalent Booster Dose Formulation)(BLUE CAP)\x11\x110',
#  'Pfizer Covid-19 Vaccine\x11\x110',
#  'COVID-19 Moderna mRNA 12 yrs and older\x11\x110',
#  ' Pfizer COVID-19 Vaccine(Purple Top) 12 Years and up\x11\x110',
#  'COVID-19 (Pfizer) mRNA, LNP-S, PF, 30 mcg/0.3 mL dose (CVX-208)\x11\x110',
#  '#Pfizer COVID-19, SARS-COV2 vaccine\x11\x110',
#  'COVID-19 mRNA LNP-S, bivalent PF 12yrs-adult booster (Pfizer), 30mcg/0.3mL, external admin\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (18y+Pri/Bstr)-MOD\x11\x110',
#  'COVID-19 (PFIZER, MAROON CAP) 6 MO-4 YRS, 0.2 ML (DOSES 1 & 2)\x11\x110',
#  '                         SARS-COV-2 (COVID-19) vaccine, mRNA, spike protein, LNP, preservative free, 30 mcg/0.3mL dose                       \x11\x110',
#  'Pfizer COVID-19 vaccine 18yrs+, preservative free, tris-sucrose formulation 30 mcg/0.3mL dose, 12 years and older (Pfizer)\x11\x110',
#  'Covid-19 (Moderna)\x11\x110',
#  'SARS-CoV-2 (aka Janssen J&J)\x11\x110',
#  "COVID BOOSTER'S \x11\x110",
#  'Pfizer SARS-CoV-2 30mcg/0.3mL Purple Cap\x11\x110',
#  'Covid 19 Vaccine, Unspecified\x11\x110',
#  'COVID-19, Pfizer Purple top, DILUTE for use, 12+ yrs, 30mcg/0.3mL dose\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 3 mcg/0.2 mL dose, tris-sucrose\x11\x110',
#  'COVID-19 Moderna 25 mcg/0.25 mL (Pediatric) Booster dose: BIVALENT VACCINE (Blue capped vial with gray-bordered label) 6-11yrs\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi1\x11\x110',
#  'COVID-19 PF (Janssen/J&J), Patient Reported / Not Verified\x11\x110',
#  'Moderna (Spikevax) COVID-19\x11\x110',
#  'COVID-19 PFIZER Booster\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 30 mcg/0.3 mL dose (Pfizer-BioNTech)\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, unspecifi2\x11\x110',
#  'COVID-19 PF Vaccine 0.5ml (Janssen)\x11\x110',
#  'SARS-CoV-2 mRNA tozinameran-tris-sucr\x11\x110',
#  'COVID-19 mRNA LNP-S PF 50 mcg/0.25 mL Monovalent Booster (Moderna)\x11\x110',
#  'Moderna Sars-cov-2 Bivalent Vaccination 18+ y.o.(Grey)\x11\x110',
#  'Pfizer Covid-19, (12 yrs and older) DILUTION REQUIRED\x11\x110',
#  'Pfizer Purple Cap SARS_COV-2 Vaccination\x11\x110',
#  'Novavax Sars-cov-2 Vaccination\x11\x110',
#  'COVID-19 Johnson & Johnson\x11\x110',
#  'COVID-19 Johnson & Johnson Booster\x11\x110',
#  'Pfizer COVID 19 Vaccine\x11\x110',
#  'COVID-19 (JANSSEN), PF\x11\x110',
#  '#Moderna Omicron Bivalent Booster 12+ Years\x11\x110',
#  'COVID-19, 3RD DOSE, Pfizer BOOSTER\x11\x110',
#  'COVID-19 PF (Janssen/J&J)\x11\x110',
#  'Covid-19 Mrna Lnp-s PF 30 Mcg/0.3 Ml Im Monovalent Booster (Pfizer)\x11\x110',
#  'SARS-CoV-2 Moderna\x11\x110',
#  'Moderna SARS-CoV-2\x11\x110',
#  'COVID-19 booster vaccine, age 12+ yr, bivalent (PFIZER-BIONTECH)\x11\x110',
#  'COVID-19 Moderna, 100 or 50 mcg\x11\x110',
#  'SARS-COV-2 for 18+ YO 0.25 mL BOOSTER (Moderna Red Label)\x11\x110',
#  '#Moderna COVID-19, SARS-COV2 vaccine\x11\x110',
#  'COVID-19 Sinopharm inactivated, non-US (BIBP)',
#  'SARS-COV-2 (COVID-19) vax, unspecified\x11\x110',
#  'SARS-CoV-2, Unspecified\x11\x110',
#  'COVID-19 MRNA (PFIZER 5-11 YRS) 10 MCG/0.2 ML TRIS-SUCROSE\x11\x110',
#  'SARS-CoV-2 mRNA (tozinameran 5y-11y)\x11\x110',
#  'Pfizer Sars-cov-2 Vaccine (Purple Cap)\x11\x110',
#  'Moderna SARS-CoV-2 Vaccination (18 + y/o)(Monovalent Booster Half-Dose)(RED-CAP)\x11\x110',
#  'Covid-19 Vaccine, mRNA, LNP-S, Pf, Diluent Reconstituted, Age 12+(Pfizer-BioNTech)\x11\x110',
#  'Moderna COVID 19 Vaccine\x11\x110',
#  'SARS-CoV-2 mRNA-1273 bivalent boost vax\x11\x110',
#  'COVID-19 mRNA LNP-S PF 100 mcg/0.5 mL dose (Moderna) 18+yrs\x11\x110',
#  'Moderna COVID-19 Vaccine, red cap blue label, Adult (18+) Booster, 12+ Primary Series\x11\x110',
#  'COVID-19 VACC,MRNA,(PFIZER)(PF)(IM)\x11\x110',
#  'Moderna Covid-19, Full Dose or Immunocompromised 3rd dose\x11\x110',
#  'COVID-19 - Johnson & Johnson vaccine, vector-nr, rS-Ad26, PF, 0.5 mL\x11\x110',
#  'COVID-19 mRNA LNP-S PF 10 mcg/0.2 mL Bivalent Booster (Pfizer) 5-11YRS\x11\x110',
#  'Moderna SARS-COV-2\x11\x110',
#  'Sars-cov-2 (Covid-19) Vaccine Unspecified\x11\x110',
#  'SARS-COV-2 (COVID-19) VACCINE, MODERNA BOOSTER\x11\x110',
#  'Moderna SARS-CoV-2 Booster Vaccination\x11\x110',
#  '#Pfizer Omicron Bivalent Booster 12+ Years\x11\x110',
#  'Covid-19 Vaccine 18yr+ (Janssen)(IMM609)\x11\x110',
#  'COVID-19, mRNA, (Pfizer - Purple Cap) 30 mcg/0.3 mL\x11\x110',
#  'MODERNA, 100 MCG, SARS-COV2 (COVID-19) VACCINE\x11\x110',
#  'Janssen/J&J COVID-19 Vaccine INJ\x11\x110',
#  'COVID-19\x11\x110',
#  'Covid-19 Vaccine MRNA(PF, Premixed) 12yr+ (PFIZER/BIONTECH)(IMM613)\x11\x110',
#  'COVID-19 Vaccine (Pfizer) Monovalent (Purple Top)\x11\x110',
#  'COVID-19 Pfizer mRNA 12 yrs and older (purple cap)\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose, tris-sucrose (Pfizer-BioNTech)\x11\x110',
#  'Moderna SARS-CoV-2 Vaccination Booster\x11\x110',
#  'COVID-19 Vaccine Moderna 1st dose\x11\x110',
#  'Pfizer SARS-CoV-2 Bivalent Vaccination\x11\x110',
#  'Moderna\x11\x110',
#  'Covid-19 mRNA 12 Yrs+ Preservative Free (Pfizer)\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 50 mcg/0.5 mL dose\x11\x110',
#  'Pfizer SARS-CoV-2 (PURPLE CAP) 12 And Older\x11\x110',
#  'Pfizer Covid-19, Bivalent Booster (12+ Yr Old)\x11\x110',
#  'SARS-CoV-2 (COVID-19) mRNA BNT-162b2 vax2\x11\x110',
#  'SARS-CoV-2 mRNA (tozinameran) vaccine1\x11\x110',
#  'MODERNA COVID-19 VACCINE (GREY LABEL/BLUE CAP), 12+ BIVALENT BOOSTER\x11\x110',
#  'Covid Vaccine Pfizer-Biontech\x11\x110',
#  'SARS-COV-2 (COVID-19)Booster\x11\x110',
#  'Pfizer SARS-CoV-2\x11\x110',
#  'PFIZER 12+ YR PURPLE CAP (must dilute) SARS-COV-2 (COVID-19) vaccine, mRNA, 30mcg/0.3mL dose\x11\x110',
#  'COVID-19 MRNA LNP-S PF 30MCG/0.3 ML IM PFIZER (TPC)\x11\x110',
#  'COVID-19 (MODERNA) BIVALENT BOOSTER 12 yrs+ 0.5 mL\x11\x110',
#  'COVID-19(Pfizer) mRNA, LNP-S, PF, 30 mcg/0.3 mL Vaccine\x11\x110',
#  'COVID-19 MRNA (PFIZER) 30 MCG/0.3 ML TRIS-SUCROSE\x11\x110',
#  'SARS-CoV-2 (C-19) mRNA (12y+ bi-B)-PFR\x11\x110',
#  'Pfizer (Diluent Reconstituted) COVID19 Vaccine, 0.3mL per dose, 2 doses, administered 21 days apart\x11\x110',
#  'COVID-19 Pfizer 30 mcg/0.3 mL Booster dose: BIVALENT VACCINE (Gray capped vial with gray-bordered label) 12+yrs (TPC)\x11\x110',
#  '(Booster) Pfizer: SARS-CoV-2 (COVID-19)\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ4\x11\x110',
#  'SARS-CoV-2 mRNA (tozinameran) vaccine2\x11\x110',
#  'COVID19 Janssen\x11\x110',
#  'COVID-19 NON-US VACCINE (ASTRAZENECA,COVIDSHIELD,VAXZEVRIA)\x11\x110',
#  'COVID-19 Booster vaccine >12 years (Pfizer-Biontech, Bivalent) IM Injection 30 mcg/0.3mL\x11\x110',
#  'COVID-19 PFizer-BioNtech 12yrs-adult, external administration\x11\x110',
#  'Pfizer,Purple Cap,Covid-19,mrna,lnp-s,pf,0.3ml (Imm10223)\x11\x110',
#  'COVID-19, mRNA, LNP-S, Bivalent 30 mcg/0.3 mL\x11\x110',
#  'Covid-19 Vaccine MRNA (PF) 5-11yrs (Pfizer/BioNTech)(IMM611)\x11\x110',
#  'COVID-19/MODERNA Booster\x11\x110',
#  'COVID 12+YO MONOVALENT Moderna 100/0.5 207\x11\x110',
#  'COVID 12+YO BIVALENT Pfizer 30/0.3 300\x11\x110',
#  'SARS-CoV-2 (COVID-19) Vaccine, Unspecified\x11\x110',
#  'SARS-CoV-2 (COVID-19) vaccine (Pfizer), mRNA, LNP-S, PF, 30 mcg/0.3 mL dose\x11\x110',
#  'SARS-COV-2 COVID-19 MODERNA 12+ YRS VACCINE\x11\x110',
#  'Bivalent booster,(Moderna) preservative free, 50 mcg/0.5 mL or 25 mcg/0.25 mL dose (Moderna)\x11\x110',
#  'Covid 19\x11\x110',
#  'Moderna COVID-19 vaccine Booster, preservative free, 50 mcg/0.25 mL dose (Moderna)\x11\x110',
#  '                       SARS-COV-2 (COVID-19) vaccine, UNSPECIFIED                     \x11\x110"',
#  'MODERNA COVID-19 VACCINEBIVALENTBA.4BA.5 50MCG0 INJ MODE\x11\x110',
#  'Moderna Booster SARS-COV-2 (COVID-19) Vaccine\x11\x110',
#  'Moderna (Spikevax) COVID-19, 12+ Yrs\x11\x110',
#  'MODERNA SARS-COV-2 Bivalent Booster 6+\x11\x110',
#  'SARS-CoV-2 (C-19) mRNA (bi-B) -MOD\x11\x110',
#  'COVID-19,SARS-COV-2 VACCINE, UNSPECIFIED\x11\x110',
#  'Pfizer SARS-CoV-2 Vaccination (Gray Cap)\x11\x110',
#  'MODERNA SARS-COV-2 COVID-19 VACCINE 0.5ML\x11\x110',
#  'COVID-19 vaccine (Pfizer-BioNTech 30mcg/0.3mL) 12YO+ BIVALENT BOOSTER PF, MDV\x11\x110',
#  'COVID-19 (J & J/Janssen)\x11\x110',
#  'Covid-19 Vaccine (Pf) (Janssen)-IMM609\x11\x110',
#  'Moderna COVID-19 vaccine, preservative free, 100 mcg/0.5mL dose (Moderna)\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100mcg/0.5 mL dose\x11\x110',
#  'COVID-19 VACCINE\x11\x110',
#  'Pfizer Sars-cov-2 Bivalent Vaccination 12+ y.o.(Grey)\x11\x110',
#  'SARS-CoV-2 mRNA (tozinameran) vaccine3\x11\x110',
#  'SARS-COV-2 mRNA (tozinameran 12y+) bival\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 10 mcg/0.2 mL dose, tris-sucrose (Pfizer-BioNTech)\x11\x110',
#  'COVID-19 original vaccine, full dose, monovalent (MODERNA)\x11\x110',
#  'Pfizer COVID-19 Vaccine EUA 30 mcg/0.3 mL\x11\x110',
#  'Pfizer Covid Vaccine, Bivalent (12+)\x11\x110',
#  'Moderna SARS-CoV-2 Booster Vaccination (50 mcg/0.25 mL)\x11\x110',
#  'COVID-19 MRNA LNP-S PF 50mcg/0.25mL Monvalent Booster (Moderna)(TPC)\x11\x110',
#  'SARS-COV-2 (COVID-19) - MODERNA\x11\x110',
#  'Moderna Covid-19, Half Dose (Booster or Immunocompromised 4th dose)\x11\x110',
#  'ConSina (COVID-19)  for historical entry\x11\x110',
#  'Moderna Sars-cov-2 Vaccine\x11\x110',
#  'COVID-19 MRNA (ASTRAZENECA) PF 0.5 ML\x11\x110',
#  'SARS-CoV-2 (COVID-19) vax, Moderna\x11\x110',
#  'Janssen SARS-CoV-2 Vaccination\x11\x110',
#  'Covid 19 mRNA LNP S bivalent booster PF\x11\x110',
#  'Moderna COVID-19 Vaccine Bivalent Booster 12+ yrs (NDC 0282-05)\x11\x110',
#  'COVID-19 Pfizer 30 mcg/0.3 mL Primary series: MONOVALENT VACCINE (Gray capped vial with gray-bordered label) 12+yrs (TPC)\x11\x110',
#  'COVID-19 (Moderna) mRNA 50mcg/0.5ml Bivalent Booster (12+) - (DARK BLUE CAP)\x11\x110',
#  'COVID-19 Pfizer\x11\x110',
#  'SARS-COV-2 (COVID-19) vaccine, vector non-replicating, recombinant spike protein-Ad26, preservative free, 0.5 mL (Janssen)\x11\x110',
#  'Janssen COVID19 Johnson Johnson\x11\x110',
#  'COVID-19 Moderna 100 mcg/0.5 mL\x11\x110',
#  'Covid-19 (Pfizer)\x11\x110',
#  'COVID-19 mRNA LNP-S PF 25 mcg/0.25 mL Bivalent Booster (Moderna) 6-11yrs\x11\x110',
#  'Pfizer SARS-COV-2 (COVID-19) Vaccine, 30mcg/0.3ml dose\x11\x110',
#  'SARS-COV-2 (COVID-19) Vaccine (mRNA, Spike Protein, LNP, Preservative Free, 100 mcg/0.5 mL or 50 mcg/0.25 mL dose)\x11\x110',
#  'Pfizer-BioNTech COVID-19 Vaccine mRNA\x11\x110',
#  'MODERNA COVID-19 VACCINE 100 MCG0.5ML INJ MODE\x11\x110',
#  'AstraZeneca Covid-19 Vaccine\x11\x110',
#  'COVID-19 Pfizer, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose, Comirnaty (gray cap) (CVX=217)\x11\x110',
#  'SARS-CoV-2(C-19) mRNA (12y+ purple)-PFZ2\x11\x110',
#  'SARS-CoV-2 mRNA(tozinameran) vaccine-Pfz1\x11\x110',
#  'COVID-19 (MODERNA) BIVALENT BOOSTER 6-11 YRS 0.25 ML\x11\x110',
#  'COVID-19, Pfizer\x11\x110',
#  'SARS-CoV-2(C-19) mRNA-1273 vaccine- Mod2\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent, PF, 50 mcg/0.5 mL dose (Moderna) - ML\x11\x110',
#  'COVID-19 Moderna 50 mcg/0.5 mL Booster dose: BIVALENT VACCINE (Blue capped vial with gray-bordered label) 12+yrs (TPC)\x11\x110',
#  '#Janssen COVID-19, SARS-COV2, 18+ yrs old, vaccine\x11\x110',
#  'COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5mL dose or 5\x11\x110',
#  'COVID-19, mRNA, LNP-S, bivalent booster, PF, 30 mcg/0.3 mL dose (Pfizer-BioNTech) - ML\x11\x110',
#  '(PFIZER) (12 YR UP) COVID-19 VACCINE - EMERGENCY USE AUTHORIZATION, MRNA,BNT162B2(PF) 30 MCG/0.3 ML IM SUSP\x11\x110',
#  'Moderna Sars-Cov-2 Vaccine\x11\x110',
#  'Moderna Covid Booster (Half Dose)\x11\x110',
#  'COVID-19 (Moderna) mRNA 100 MCG MDV Vaccine Primary Series and Immunocompromised\x11\x110',
#  'COVID-19 PFIZER (12+ YRS)\x11\x110',
#  'Pfizer SARS-CoV-2 Vaccination, age 12+ (Purple Top)\x11\x110',
#  'Pfizer COVID vaccine, orange cap, 5-11\x11\x110',
#  'COVID-19 12+YO Moderna/18+YO Boost 100 mcg/0.5 mL (RED CAP)\x11\x110',
#  'COVID-19 Moderna Vaccine (1st,2nd,3rd dose = 0.5ml)\x11\x110',
#  'COVID-19 Pfizer Monovalent Vaccine\x11\x110',
#  'ZZModerna Covid-19 BOOSTER 0.25ml\x11\x110',
#  'COVID-19 vaccine, Subunit, rS-nanoparticle+Matrix-M1 Adjuvant, PF, 0.5 mL (Novavax)\x11\x110',
#  'COVID-19, subunit, rS-nanoparticle+Matrix-M1 Adjuvant, PF, 0.5 mL\x11\x110']
