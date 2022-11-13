#%% [markdown]
# ## Explore dataset
#%%
import pandas as pd
from pathlib import Path

# Load Data
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
cohort_details_path = data_dir / "raw/v1/Cohort Details_9602.csv"
diagnosis_path = data_dir / "raw/v1/Results_DiagnosesCleanedAggregated_9602.csv"
labs_path = data_dir / "raw/v1/Results_Laboratories_9602.csv"
case_path = data_dir / "raw/v1/Results_Main_Case_9602.csv"
summary_path = data_dir / "raw/v1/Summary_9602.csv"

cohort_df = pd.read_csv(cohort_details_path)
diagnosis_df = pd.read_csv(diagnosis_path)
labs_df = pd.read_csv(labs_path)
case_df = pd.read_csv(case_path)
summary_df = pd.read_csv(summary_path)
# %%
# Cohort Info
print("Cohort Info Shape: ", cohort_df.shape)
cohort_df
#%%
# Summary Info - Info on the Columns for Cases
print("Summary Info Shape: ", summary_df.shape)
summary_df
# %%
# Diagnosis Info - List of ICD 10 Diagnosis for CaseID & PatientID
print("Diagnosis Shape: ", diagnosis_df.shape)
#%%
# Labs - List of labs for CaseID & PatientID
print("Labs Shape: ", labs_df.shape)
#%%
# Case Info - Main Dataframe with 84 columns
print("Case Shape: ", case_df.shape)
print(f"Case Columns: {'; '.join(case_df.columns)}")
# Case Shape:  (37230, 84)
# Case Columns: MPOG_Case_ID; MPOG_Patient_ID; Institution_Name; AdmissionType_Value;
# AdmissionType_Value_Code; AgeInYears_Value; AKI01_Result; AKI01_Result_Reason;
# AKI01RiskAdjusted_Confidence_Interval_High; AKI01RiskAdjusted_Confidence_Interval_Low;
# AKI01RiskAdjusted_Value; AKI01RiskAdjusted_Was_Defaulted;
# AKIProgressionRisk_Value; AnesthesiaDuration_Value;
# AnesthesiaTechniqueGeneral_Value; AnesthesiaTechniqueGeneral_Value_Code;
# AnesthesicGasesUsedHalogenated_Value; AnesthesicGasesUsedHalogenated_Value_Code;
# AsaStatusClassification_Value; AsaStatusClassification_Value_Code;
# BodyMassIndex_Value; CARD02_Result; CARD02_Result_Reason;
# ComorbidityElixhauserCardiacArrhythmias_Value;
# ComorbidityElixhauserChronicPulmonaryDisease_Value;
# ComorbidityElixhauserCongestiveHeartFailure_Value;
# ComorbidityElixhauserDiabetesComplicated_Value;
# ComorbidityElixhauserDiabetesUncomplicated_Value;
# ComorbidityElixhauserHypertensionComplicated_Value;
# ComorbidityElixhauserHypertensionUncomplicated_Value;
# ComorbidityElixhauserLiverDisease_Value;
# ComorbidityElixhauserMetastaticCancer_Value;
# ComorbidityElixhauserObesity_Value;
# ComorbidityElixhauserPeripheralVascularDisorders_Value;
# ComorbidityElixhauserPulmonaryCirculationDisorders_Value;
# ComorbidityElixhauserRenalFailure_Value;
# ComorbidityElixhauserValvularDisease_Value;
# ComorbidityMpogCerebrovascularDisease_Value;
# ComorbidityMpogCerebrovascularDisease_Value_Code;
# ComorbidityMpogCoronaryArteryDisease_Value;
# ComorbidityMpogCoronaryArteryDisease_Value_Code;
# ComplicationAHRQCardiac_Value; ComplicationAHRQInfection_Value;
# ComplicationAHRQMyocardialInfarction_Value;
# ComplicationAHRQPulmonaryAll_Triggering_AHRQ_Diagnoses;
# ComplicationAHRQPulmonaryAll_Triggering_MPOG_Diagnoses;
# ComplicationAHRQPulmonaryAll_Value; ComplicationAHRQPulmonaryAll_Value_Code;
# ComplicationAHRQPulmonaryOther_Value; ComplicationAHRQPulmPneumonia_Value;
# ComplicationAHRQPulmRespInsuff_Value; ComplicationMpogAcuteKidneyInjury_Value;
# AnesthesiaStart_Value; EmergencyStatusClassification_Value;
# EmergencyStatusClassification_Value_Code; Height_Value; Holiday_Holiday_Name;
# Holiday_Value; Holiday_Value_Code; HospitalMortality30Day_Value;
# HospitalMortality30Day_Value_CD; MORT01_Result; MORT01_Result_Reason;
# PacuDuration_Value; ParalyticsUsed_Value; ParalyticsUsed_Value_Code;
# ParalyticsUsedNondepolarizing_Value; ParalyticsUsedNondepolarizing_Value_Code;
# PostopLengthOfStayDays_Value; PrimaryAnesthesiaCPT_MPOGAnesCPTClass;
# PrimaryAnesthesiaCPT_MPOGbaseUnits; PrimaryAnesthesiaCPT_uploadedUnits;
# PrimaryAnesthesiaCPT_value; ProcedureTypeCardiacAlt_Source_Text;
# ProcedureTypeCardiacAlt_Value; ProcedureTypeCardiacAlt_Value_Code;
# Race_Value; Race_Value_Code; Sex_Value; Sex_Value_Code; SurgeryDuration_Value;
# VolatileGasesUsed_Value; VolatileGasesUsed_Value_Code; Weight_Value
#%% [markdown]
# ## Patient Demographics
# %%
subset_columns = [
    "MPOG_Case_ID",
    "MPOG_Patient_ID",
    "Institution_Name",
    "AgeInYears_Value",
    "Race_Value",
    "Sex_Value",
    "Weight_Value",
    "Height_Value",
    "BodyMassIndex_Value",
    "ProcedureTypeCardiacAlt_Source_Text",
    "ProcedureTypeCardiacAlt_Value",
    "ProcedureTypeCardiacAlt_Value_Code",
]
df = case_df.loc[:, subset_columns]
#%%
# Age (no missing data)
df.AgeInYears_Value.describe()
# count    37230.000000
# mean        52.574138
# std         18.449849
# min          0.166666
# 25%         37.000000
# 50%         55.000000
# 75%         67.000000
# max         90.000000
# Name: AgeInYears_Value, dtype: float64
#%%
df.Race_Value.value_counts(dropna=False)
# White, not of hispanic origin       25944
# Asian or Pacific Islander            3460
# Hispanic, white                      2692
# Black, not of hispanic origin        2520
# Unknown race                         1664
# American Indian or Alaska Native      879
# Hispanic, black                        71
# Name: Race_Value, dtype: int64``

# 1664 patient with unknown race
#%%
df.Weight_Value.describe()
# count    36791.000000
# mean        83.142714
# std         23.768729
# min          7.320000
# 25%         67.399000
# 50%         80.099000
# 75%         95.699000
# max        249.929000
# Name: Weight_Value, dtype: float64
#%%
df.Height_Value.describe()
# count    36675.000000
# mean       170.162395
# std         12.616618
# min         12.721000
# 25%        162.822000
# 50%        170.455000
# 75%        178.087000
# max        218.793000
# Name: Height_Value, dtype: float64

#%%
df.BodyMassIndex_Value.describe()
# count    36469.000000
# mean        28.558837
# std          7.303382
# min         11.290000
# 25%         23.550000
# 50%         27.350000
# 75%         32.110000
# max         78.750000
# Name: BodyMassIndex_Value, dtype: float64
#%%
# Denotes if case is a cardiac case or not
df.ProcedureTypeCardiacAlt_Value.value_counts()
# No                                      34292
# Cardiac - EP/Cath                        1965
# Cardiac - Open                            457
# Cardiac - Transcatheter/Endovascular      298
# Cardiac - Other                           218
# Name: ProcedureTypeCardiacAlt_Value, dtype: int64
#%%
# How many unique patients
len(df.MPOG_Patient_ID.unique())
# 27597
#%%
len(df.MPOG_Case_ID.unique())
# 37230
#%%
# Patients can have multiple surgeries.  How many cases for each patient?
df = df.loc[:, ["MPOG_Patient_ID", "MPOG_Case_ID"]]
num_cases_per_patient = df.groupby("MPOG_Patient_ID").count()
num_cases_per_patient = (
    num_cases_per_patient.value_counts()
    .rename("PatientCount")
    .rename_axis(index="NumCases")
)
num_cases_per_patient
# NumCases
# 1           21952
# 2            3830
# 3            1034
# 4             353
# 5             180
# 6              98
# 7              38
# 9              26
# 8              25
# 10             13
# 11             11
# 13              7
# 12              5
# 15              4
# 16              3
# 18              3
# 24              2
# 17              2
# 19              2
# 27              2
# 22              2
# 38              1
# 14              1
# 21              1
# 20              1
# 41              1
# Name: PatientCount, dtype: int64
#%%
num_cases_per_patient.plot()

#%% [markdown]
# ## Admission & Mortality Info
# %%
# Get simplified cases df
subset_columns = [
    "MPOG_Case_ID",
    "MPOG_Patient_ID",
    "Institution_Name",
    "AdmissionType_Value",
    "HospitalMortality30Day_Value",
    "PostopLengthOfStayDays_Value",
    "PacuDuration_Value",
    "AnesthesiaDuration_Value",
    "SurgeryDuration_Value",
]
df = case_df.loc[:, subset_columns]

#%%
df.Institution_Name.value_counts()
# University of Washington Medical Center - IM    37230
# Name: Institution_Name, dtype: int64

# NOTE: even though data from 3 hospitals, they are all grouped under same institution--separate field for hospital name?

# %%
df.AdmissionType_Value.value_counts()
# Inpatient                 18050
# Outpatient                16953
# 23 hour observation        1702
# Emergency                   228
# Other Admission Type        148
# Admit                       147
# Unknown Concept               1
# Unknown Admission Type        1
# Name: AdmissionType_Value, dtype: int64

# NOTE: will need to clean/drop values
# %%
df.HospitalMortality30Day_Value.value_counts()
# No             36612
# Yes              577
# Conflicting       41
# Name: HospitalMortality30Day_Value, dtype: int64

# TODO: what does "conflicting" mean?  For now, we will need to drop these cases.
#%% [markdown]
# ## Length of Stay Metrics
#%%
# PACU Length of Stay (presumed in minutes)
# NOTE: there is some invalid negative values that we need to drop/clean
lower_bound = 0
upper_bound = 500
lower_outliers = df.PacuDuration_Value.loc[df.PacuDuration_Value < lower_bound]
# 90      -999
# 145     -999
# 214     -999
# 231     -999
# 329     -999
#         ...
# 37103   -999
# 37105   -999
# 37151   -999
# 37158   -999
# 37166   -999
# Name: PacuDuration_Value, Length: 376, dtype: int64
upper_outliers = df.PacuDuration_Value.loc[df.PacuDuration_Value > upper_bound]
# 15177     780
# 23144     563
# 29545     903
# 36032    1041
# Name: PacuDuration_Value, dtype: int64

# Truncate data to between (lower, upper), then visualize PACU length of stay
pacu_duration = df.PacuDuration_Value.loc[
    (df.PacuDuration_Value > lower_bound) & (df.PacuDuration_Value < upper_bound)
]
pacu_duration.hist(bins=100)
#%%
# Statistics on PACU Length of Stay
pacu_duration.describe()
# count    35299.000000
# mean        92.215219
# std         78.432601
# min          1.000000
# 25%         34.000000
# 50%         73.000000
# 75%        124.000000
# max        405.000000
# Name: PacuDuration_Value, dtype: float64

# Presume the units for PACU Length of Stay data field is [minutes]


#%%
# 10% of data is misisng post-op LOS data
df.PostopLengthOfStayDays_Value.isna().value_counts()
# False    33878
# True      3352
# Name: PostopLengthOfStayDays_Value, dtype: int64
#%%
# Post-op Length of Stay for all cases
df.PostopLengthOfStayDays_Value.describe()
# count    33878.000000
# mean         5.308489
# std         13.831038
# min          0.000000
# 25%          0.000000
# 50%          1.000000
# 75%          4.000000
# max        255.000000
# Name: PostopLengthOfStayDays_Value, dtype: float64

#%%
# Visualize as Histogram
df.PostopLengthOfStayDays_Value.hist(bins=100)
#%%
## Long tail, so let's truncate data to <30 days
upper_bound = 30
df_less_than_30 = df.PostopLengthOfStayDays_Value[
    df.PostopLengthOfStayDays_Value < upper_bound
]
df_less_than_30.hist(bins=100)
#%%
# Surgery Duration
# Many surgeries don't have recorded durtion (~20%)
df.SurgeryDuration_Value.isna().value_counts()
# False    28576
# True      8654
# Name: SurgeryDuration_Value, dtype: int64
#%%
# Statistics on Surgery Duration
lower_bound = 0
surgery_duration = df.SurgeryDuration_Value[df.SurgeryDuration_Value.notna()]
surgery_duration = df.SurgeryDuration_Value[df.SurgeryDuration_Value > lower_bound]
surgery_duration.describe()
# count    28565.000000
# mean       122.392963
# std        113.119341
# min          1.000000
# 25%         45.000000
# 50%         87.000000
# 75%        160.000000
# max        969.000000
# Name: SurgeryDuration_Value, dtype: float64
#%%
surgery_duration.hist(bins=100)
#%%
# Statistics on Anesthesia Duration
lower_bound = 0
anesthesia_duration = df.AnesthesiaDuration_Value[df.AnesthesiaDuration_Value.notna()]
anesthesia_duration = df.AnesthesiaDuration_Value[
    df.AnesthesiaDuration_Value > lower_bound
]
anesthesia_duration.describe()
# count    37230.000000
# mean       180.162772
# std        174.412499
# min          5.000000
# 25%         72.000000
# 50%        131.000000
# 75%        227.000000
# max       2125.000000
# Name: AnesthesiaDuration_Value, dtype: float64
#%%
anesthesia_duration.hist(bins=100)
# Some cases have way longer anesthesia duration than surgery duration

# %%
# TODO: figure out how to get COVID19 diagnosis out of diagnosis table.  Which ICD10 code.

#%%
# TODO: figure out how to join labs table with patient table
#%%
# Possible result values
labs_df.AIMS_Value_Text.value_counts().to_frame()
# TODO: The value field is very messy--we need to harmonize this
# 	AIMS_Value_Text
# None detected	97803
# Detected	1807
# Duplicate request	95
# NEGATIVE	81
# Inconclusive	49
# Cancel order changed	34
# Reorder requested label error	29
# Wrong test ordered by practitioner	23
# NEG	15
# POSITIVE	14
# Canceled by practitioner	8
# Follow-up testing required. Sample recollection requested.	8
# Specimen not labeled	5
# Negative	5
# Reorder requested improper tube/sample type	3
# Data entry correction see updated information	3
# Reorder requested sample lost	3
# Not detected	2
# Reorder requested sample problem	2
# Wrong test selected by UW laboratory	2
# neg	2
# Reorder requested. No sample received.	1
# Follow-up testing required. Refer to other SARS-CoV-2 Qualitative PCR result on specimen with similar collection date and time.	1
# POS	1
# detected	1
# Cancel see detail	1
# None Detected	1
# negative	1
#%%
# Lets clean these values and then get an accurate count

def clean_covid_result_value(value: str):
    positive_values = ["Detected", "POSITIVE", "POS", "detected"]
    negative_values = ["None detected", "NEGATIVE", "NEG", "Negative", "Not detected", "neg", "None Detected", "negative"]
    unknown_values = [
        "Inconclusive",
        "Duplicate request",
        "Cancel order changed",
        "Reorder requested label error",
        "Wrong test ordered by practitioner",
        "Canceled by practitioner",
        "Follow-up testing required. Sample recollection requested.",
        "Specimen not labeled",
        "Reorder requested improper tube/sample type",
        "Data entry correction see updated information",
        "Reorder requested sample lost",
        "Reorder requested sample problem",
        "Wrong test selected by UW laboratory",
        "Reorder requested. No sample received.",
        "Follow-up testing required. Refer to other SARS-CoV-2 Qualitative PCR result on specimen with similar collection date and time.",
        "Cancel see detail"
    ]
    if value.lower() in [x.lower() for x in positive_values]:
        return "Positive"
    elif value.lower() in [x.lower() for x in negative_values]:
        return "Negative"
    elif value.lower() in [x.lower() for x in unknown_values]:
        return "Unknown"
    else:
        raise ValueError(f"Unknown value {value} encountered that is not handled by clean_covid_result_value() logic.")

cleaned_covid_result_values = labs_df.AIMS_Value_Text.apply(clean_covid_result_value)
cleaned_covid_result_values.value_counts()
# Negative    97910
# Positive     1823
# Unknown       267
# Name: AIMS_Value_Text, dtype: int64


#%%
# Get table with each row = patient, each case is a column with value of Case ID & COVID test result
labs_df.groupby("MPOG_Patient_ID").apply(
    lambda x: list(zip(x.MPOG_Case_ID, x.AIMS_Value_Text))
).apply(pd.Series)
# %%
