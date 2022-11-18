#%% [markdown]
# ## Explore dataset
#%%
import pandas as pd
from datetime import datetime, timedelta
import hashlib
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
cohort_df.head()
#%%
# Summary Info - Info on the Columns for Cases
print("Summary Info Shape: ", summary_df.shape)
summary_df.head()
# %%
# Diagnosis Info - List of ICD 10 Diagnosis for CaseID & PatientID
print("Diagnosis Shape: ", diagnosis_df.shape)
diagnosis_df.head()
#%%
# Labs - List of labs for CaseID & PatientID
print("Labs Shape: ", labs_df.shape)
labs_df.head()
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
case_df.head()
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
num_cases_per_patient.plot(title="Number of Cases per Patient")

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
pacu_duration.plot(kind="hist", bins=100, title="Histogram of PACU Duration")
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
df.PostopLengthOfStayDays_Value.plot(
    kind="hist", bins=100, title="Histogram of Post-op Length of Stay Days"
)
#%%
## Long tail, so let's truncate data to <30 days
upper_bound = 30
df_less_than_30 = df.PostopLengthOfStayDays_Value[
    df.PostopLengthOfStayDays_Value < upper_bound
]
df_less_than_30.plot(
    kind="hist", bins=100, title="Histogram of Post-op Length of Stay Days"
)
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
surgery_duration.plot(kind="hist", bins=100, title="Histogram of Surgery Duration")
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
anesthesia_duration.plot(
    kind="hist", bins=100, title="Histogram of Anesthesia Duration"
)
# Some cases have way longer anesthesia duration than surgery duration

# %% [markdown]
# ## Diagnosis
# TODO: figure out how to get COVID19 diagnosis out of diagnosis table.  Which ICD10 code.

# %%
# [markdown]
# ## Vaccine Data
# TODO: COVID vaccine data for patients?  Where can we get this?

#%% [markdown]
# ## Explore COVID Labs Table
#%%
# Possible result values
labs_df.AIMS_Value_Text.value_counts().to_frame()

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
    negative_values = [
        "None detected",
        "NEGATIVE",
        "NEG",
        "Negative",
        "Not detected",
        "neg",
        "None Detected",
        "negative",
    ]
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
        "Cancel see detail",
    ]
    if value.lower() in [x.lower() for x in positive_values]:
        return "Positive"
    elif value.lower() in [x.lower() for x in negative_values]:
        return "Negative"
    elif value.lower() in [x.lower() for x in unknown_values]:
        return "Unknown"
    else:
        raise ValueError(
            f"Unknown value {value} encountered that is not handled by clean_covid_result_value() logic."
        )


def create_uuid(data: str, format: str = "T-SQL") -> str:
    """Creates unique UUID using BLAKE2b algorithm.

    Args:
        data (str): input data used to generate UUID
        format (str): Output format.
            `None` results in raw 32-char digest being returned as UUID.
            `T-SQL` results in 36-char UUID string (32 hex values, 4 dashes)
                delimited in the same style as `uniqueidentifier` in T-SQL databases.

    Returns:
        Formatted UUID.
    """
    data = data.encode("UTF-8")
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()
    if format is None:
        return digest
    elif format == "T-SQL":
        uuid = (
            f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:]}"
        )
        return uuid
    else:
        raise ValueError(f"Unknown argument {format} specified for `format`.")


def format_labs_df(labs_df: pd.DataFrame):
    # Create UUID for each Lab Value based on MPOG Patient ID, Lab Concept ID, Lab DateTime.
    # If there are any duplicated UUIDS, this means the row entry has the same value for all 3 of these.
    labs_df["LabUUID"] = labs_df.apply(
        lambda row: create_uuid(
            row.MPOG_Patient_ID
            + str(row.MPOG_Lab_Concept_ID)
            + row.AIMS_Lab_Observation_DT
        ),
        axis=1,
    )
    # Drop "AIMS_Value_Numeric" and "AIMS_Value_CD" columns because they have no
    # information and are all `nan` values.
    # Drop "MPOG_Lab_Concept_ID" and "Lab_Concept_Name" because this table is only
    # Lab_Concept_Name='Virology - Coronavirus (SARS-CoV-2)', MPOG_Lab_Concept_ID=5179\
    labs_df = labs_df.drop(
        columns=[
            "AIMS_Value_Numeric",
            "AIMS_Value_CD",
            "MPOG_Lab_Concept_ID",
            "Lab_Concept_Name",
        ]
    )
    # Clean COVID result column
    labs_df.AIMS_Value_Text = labs_df.AIMS_Value_Text.apply(clean_covid_result_value)
    # Format String date into DateTime object
    labs_df.AIMS_Lab_Observation_DT = labs_df.AIMS_Lab_Observation_DT.apply(
        lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
    )
    return labs_df


df = format_labs_df(labs_df)

# Breakdown of COVID Lab Results
df.AIMS_Value_Text.value_counts()
# Negative    97910
# Positive     1823
# Unknown       267
# Name: AIMS_Value_Text, dtype: int64
# %%
# Number of COVID Lab Results by Month
df.AIMS_Lab_Observation_DT.apply(lambda dt: dt.month).value_counts().sort_index(
    ascending=True
).plot(kind="bar", title="Number of COVID Lab Results by Month")

#%%
# Number of Positive, Negative, Unknown COVID Lab Results by Month
results_by_date = df.loc[:, ["AIMS_Lab_Observation_DT", "AIMS_Value_Text"]]
results_by_date.AIMS_Lab_Observation_DT = results_by_date.AIMS_Lab_Observation_DT.apply(
    lambda dt: dt.month
)
results_by_date.groupby("AIMS_Lab_Observation_DT").value_counts().plot(
    kind="bar", title="Number of Positive, Negative, Unknown COVID Lab Results by Month"
)
#%% Number of Positive, Nevative, Unknown COVID Lab Results by Month
results_by_date.groupby("AIMS_Lab_Observation_DT").value_counts().unstack(
    -1
).rename_axis(index="Month")
#%% Percent of Positive, Nevative, Unknown COVID Lab Results by Month
results_by_date.groupby("AIMS_Lab_Observation_DT").value_counts(normalize=True).apply(
    lambda x: f"{x:.2%}"
).unstack(-1).rename_axis(index="Month")

# %%
# Number of Positive COVID Results by Month
results_by_date.groupby("AIMS_Value_Text").value_counts().unstack(-1).loc[
    "Positive", :
].plot(kind="bar", title="Number of Positive COVID Results by Month")

# +COVID tests distribution: 600+ in Jan, 300+ in Feb, and ~100 +COVID test per month for rest of year
#%%

# %%
# Get COVID Tests per Patient
all_labs_per_patient = df.groupby("MPOG_Patient_ID")[
    ["AIMS_Lab_Observation_DT", "AIMS_Value_Text"]
].agg(list)
all_labs_per_patient
#%%
# Get Number of COVID Tests per Patient
num_covid_tests_per_patient = all_labs_per_patient.AIMS_Value_Text.apply(len)
# Aggregate Statistics on Number of COVID Tests per Patient
num_covid_tests_per_patient.describe()
# count    14270.000000
# mean         7.007708
# std         23.186766
# min          1.000000
# 25%          1.000000
# 50%          2.000000
# 75%          5.000000
# max        884.000000
# Name: AIMS_Value_Text, dtype: float64

# NOTE: suspicious that one patient has 884 COVID tests in a year.  This is 2-3 COVID tests/day.
#%%
# How many patients have more than 100 COVID Tests
num_covid_tests_per_patient[num_covid_tests_per_patient > 100].plot(
    kind="bar", title="Patients with >100 COVID Tests", figsize=(15, 8)
)

# %%
num_covid_tests_per_patient[num_covid_tests_per_patient > 100]
# %%
# Lets look at a patient who has 884 covid tests in our dataset
patient_id = "f8a460e6-798a-ec11-8dce-3cecef1ac49f"
labs_df[labs_df.MPOG_Patient_ID == patient_id].AIMS_Lab_Observation_DT
# 8992     2022-01-10 19:51:00
# 8993     2022-01-11 23:53:00
# 8994     2022-01-15 04:24:00
# 8995     2022-01-20 16:51:00
# 8996     2022-01-21 18:12:00
#                 ...
# 98571    2022-07-18 20:00:00
# 98572    2022-07-21 09:30:00
# 98573    2022-07-28 10:55:00
# 98574    2022-08-01 13:54:00
# 98575    2022-08-07 09:33:00
# Name: AIMS_Lab_Observation_DT, Length: 884, dtype: object
# %%
len(labs_df[labs_df.MPOG_Patient_ID == patient_id].AIMS_Lab_Observation_DT.unique())
# There are only 52 unique lab times here.  Suggesting many labs may be duplicated, and we need to de-duplicate labs.

# %%
# To de-duplicate all labs per patient, we need to generate a unique UUID for each lab entry,
# then group labs by patient, then deduplicate labs by time for each patient,
# then make sure the deduplicated labs have same result.
# If they have different result, then we may technically have 2 different COVID lab tests taken
# at same time with different results... which doesn't make sense.
#%%
# Look at timing of COVID labs relative to Anesthesia Case
labs_per_case_id = df.groupby("MPOG_Case_ID")["LabUUID"].agg(list).to_frame()
simplified_cases = case_df.loc[
    :, ["MPOG_Case_ID", "AnesthesiaStart_Value", "AnesthesiaDuration_Value"]
].set_index("MPOG_Case_ID")
simplified_cases.AnesthesiaStart_Value = simplified_cases.AnesthesiaStart_Value.apply(
    lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
)
# Join Labs & Cases Info.  Index is CaseID (each row is a Case)
case_lab_df = labs_per_case_id.join(simplified_cases)
# Reformat Table to be indexed by LabUUID instead (Each row is a lab test)
lab_case_df = case_lab_df.explode("LabUUID").reset_index().set_index("LabUUID")
# Add in Lab DateTime info
lab_date_time = df.set_index("LabUUID").AIMS_Lab_Observation_DT
lab_case_df = lab_case_df.join(lab_date_time)
# Get Duration Between Anesthesia Start & Lab Result Time
lab_case_df["Duration"] = (
    lab_case_df.AnesthesiaStart_Value - lab_case_df.AIMS_Lab_Observation_DT
)
lab_case_df.loc[:, ["MPOG_Case_ID", "Duration"]]
# LabUUID                               MPOG_Case_ID	                        Duration
# 00038221-3822-68f9-79d5-a38e2c22e095	ede96310-50c5-ec11-8dd1-3cecef1ac49f	34 days 08:40:00
# 00039b9f-a75f-bac0-613f-cbbd46b3d7a5	7ebde8d2-9f89-ec11-8dce-3cecef1ac49f	286 days 20:57:00
# 000452e2-fc08-76f4-72b1-eb016a5b7684	2851de29-e98f-ec11-8dcf-3cecef1ac49f	0 days 02:47:00
# 000452e2-fc08-76f4-72b1-eb016a5b7684	2851de29-e98f-ec11-8dcf-3cecef1ac49f	0 days 02:47:00
# 000452e2-fc08-76f4-72b1-eb016a5b7684	2851de29-e98f-ec11-8dcf-3cecef1ac49f	0 days 02:47:00
# ...	...	...
# fffd6b56-563f-0a69-7a8b-29552dd998d2	f151de29-e98f-ec11-8dcf-3cecef1ac49f	5 days 05:15:00
# ffffbac8-9b1d-2ae1-68bc-f05be5da70fb	2d24d0bc-4fba-ec11-8dd0-3cecef1ac49f	358 days 22:37:00
# ffffbac8-9b1d-2ae1-68bc-f05be5da70fb	2d24d0bc-4fba-ec11-8dd0-3cecef1ac49f	358 days 22:37:00
# ffffbac8-9b1d-2ae1-68bc-f05be5da70fb	c9abd8ff-449d-ec11-8dcf-3cecef1ac49f	321 days 21:14:00
# ffffbac8-9b1d-2ae1-68bc-f05be5da70fb	c9abd8ff-449d-ec11-8dcf-3cecef1ac49f	321 days 21:14:00


# NOTE: this essentially tells us that the table of labs has duplicates in several ways.
# - LabUUID depends only on PatientID, Lab Type (e.g COVID PCR Test), and Lab DateTime...
#   so when we have duplicates for a single MPOG_Case_ID, it means that this is truly a recorded duplicate
#   value.
# - Multiple MPOG_Case_IDs have been associated with the same LabUUID since all labs for a patient gets
#   associated to each case.  If we looked at each specific case & the lab data associated with it,
#   then it is not truly duplicated in that context.  But this table has all of a patient's cases, so
#   if a patient has multiple cases, we expect to get duplicates in the table.
# - Large time Duration (and negative duration values) relative to Anesthesia Start
#   reinforces the fact that we have all of a patient's labs associated to each Case (both past & future).

# These findings mean that we'll have to be careful about uniquifying labs and joining them to our cases table
#%%
deduplicated_LabUUIDs = df.LabUUID.drop_duplicates()
len(deduplicated_LabUUIDs)
# NOTE: we started with 100k lab values.  There are only 59540 after we de-duplicate by LabUUID (Patient-LabType-DateTime).
#%%

# %% [markdown]
# ## Get Duration Categories using Last COVID test
# TODO: when we decode ICD10 diagnosis, also associate COVID diagnosis as a way of telling of patient has positive COVID
#%%
## Associate Positive COVID test with each case
# Get only Positive COVID labs
pos_covid_labs_df = df.loc[df.AIMS_Value_Text == "Positive"].set_index("LabUUID")
pos_covid_LabUUIDs = pos_covid_labs_df.index
pos_covid_LabUUIDs
# NOTE: there are 1823 positive results in our dataset of 100k COVID tests
#%%
# Filter our Lab-Case associated table by only COVID+ LabUUIDs.  We have durations (AnesthesiaStart - COVID+ PCR DateTime).
pos_covid_lab_case_df = (
    lab_case_df.loc[pos_covid_LabUUIDs]
    .reset_index()
    .set_index("MPOG_Case_ID")
    .drop_duplicates()
)
# Remove Negative Durations (COVID+ test after surgery occurred)
pos_covid_lab_case_df = pos_covid_lab_case_df.loc[
    pos_covid_lab_case_df.Duration > timedelta(0)
]

# Get only most recent result for each Surgery Case
pos_covid_lab_case_df = (
    pos_covid_lab_case_df.reset_index()
    .groupby("MPOG_Case_ID")
    .apply(lambda grp_df: grp_df.sort_values(by="Duration").iloc[0])
)
pos_covid_lab_case_df
# NOTE: we are down to 763 results if we narrow to only most recent COVID+ test per case
#%%
# Now Get Statistics on Time interval between COVID+ PCR DateTime and AnesthesiaStart
pos_covid_lab_case_df.Duration.describe()
# count                           763
# mean     54 days 13:58:53.866317169
# std      73 days 09:55:35.707764318
# min                 0 days 00:10:00
# 25%                 4 days 05:24:30
# 50%                25 days 20:54:00
# 75%                71 days 09:19:00
# max               364 days 17:54:00
# Name: Duration, dtype: object
#%%
# Now we can create buckets of duration similar to how COVIDSurg 2021 paper did it
# "Timing of surgery following SARS-CoV-2 infection: an international prospective cohort study"(https://pubmed.ncbi.nlm.nih.gov/33690889/)
# - in this paper, the categorical buckets are: 0-2 weeks, 3-4 weeks; 5-6 weeks; >7 weeks

from pandas.api.types import CategoricalDtype

categorical_type = CategoricalDtype(
    categories=["0-2_weeks", "3-4_weeks", "5-6_weeks", ">=7_weeks"], ordered=True
)


def categorize_duration(duration: timedelta) -> str:
    if duration < timedelta(weeks=3):
        return "0-2_weeks"
    elif duration >= timedelta(weeks=3) and duration < timedelta(weeks=5):
        return "3-4_weeks"
    elif duration >= timedelta(weeks=5) and duration < timedelta(weeks=7):
        return "5-6_weeks"
    else:
        return ">=7_weeks"


pos_covid_lab_case_df["DurationCategory"] = pos_covid_lab_case_df.Duration.apply(
    categorize_duration
).astype(categorical_type)
# Num of Cases in each category where we have a confirmed +COVID Test
pos_covid_lab_case_df.DurationCategory.value_counts().sort_index()
# 0-2_weeks    345
# 3-4_weeks     88
# 5-6_weeks     46
# >=7_weeks    284
# Name: DurationCategory, dtype: int64

# %%
## Now we do the same analysis for CONFIRMED Negative Cases.
# - when calculate odds ratio, we can either compute odds of COVID+ to all other cases, or for only confirmed neg cases (computed below)

# Get only Negative COVID labs
neg_covid_labs_df = df.loc[df.AIMS_Value_Text == "Negative"].set_index("LabUUID")
neg_covid_LabUUIDs = neg_covid_labs_df.index

neg_covid_lab_case_df = (
    lab_case_df.loc[neg_covid_LabUUIDs]
    .reset_index()
    .set_index("MPOG_Case_ID")
    .drop_duplicates()
)
neg_covid_lab_case_df

# Remove Negative Durations (COVID- test after surgery occurred)
neg_covid_lab_case_df = neg_covid_lab_case_df.loc[
    neg_covid_lab_case_df.Duration > timedelta(0)
]
#%%
# Get only most recent result for each Surgery Case
neg_covid_lab_case_df = (
    neg_covid_lab_case_df.reset_index()
    .groupby("MPOG_Case_ID")
    .apply(lambda grp_df: grp_df.sort_values(by="Duration").iloc[0])
)
# Now Get Statistics on Time interval between COVID- PCR DateTime and AnesthesiaStart
neg_covid_lab_case_df.Duration.describe()
# count                         16454
# mean      9 days 05:06:36.385073538
# std      38 days 23:15:54.482088675
# min                 0 days 00:01:00
# 25%                 0 days 21:10:15
# 50%                 1 days 21:33:00
# 75%                 2 days 18:15:45
# max               363 days 21:23:00
# Name: Duration, dtype: object
#%%
neg_covid_lab_case_df["DurationCategory"] = neg_covid_lab_case_df.Duration.apply(
    categorize_duration
).astype(categorical_type)
# Num of Cases in each category where we have a confirmed -COVID Test
neg_covid_lab_case_df.DurationCategory.value_counts().sort_index()
# 0-2_weeks    15588
# 3-4_weeks       95
# 5-6_weeks       73
# >=7_weeks      698
# Name: DurationCategory, dtype: int64
# %%

# NOTE: the above numbers are only approximate.  We didn't check for scenario where a patient could have both COVID+ and COVID- test prior to a surgery case
len(set(pos_covid_lab_case_df.index).intersection(set(neg_covid_lab_case_df.index)))
# %%
# NOTE: When a case is COVID+ (but many weeks ago, they can have a more recent test that is COVID-).
# our numbers here don't account for this scenario and we look at the absolute numbers here.
#
# In the COVIDSurg paper, their comparison group is "No pre-operative SARS-CoV-2" by RT-PCR. (this is all patients who are COVID unknown + negative)
#
# We may be able to do better...
# Possible Scenarios:
# - Patient is COVID unknown --> surgery
# - Patient is COVID neg --> surgery [need to define time interval for this to be meaningful.  Possibly only 0-2 weeks.  Or even more strict like within 3 days.]
# - Patient is COVID pos --> surgery [time intervals computed above]
# - Patient is COVID pos, but then COVID neg --> surgery [do we distinguish this group?] . This is not handled in the logic above.
