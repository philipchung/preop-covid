from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm.auto import tqdm
from utils import read_pandas


@dataclass
class CaseData:
    """Data object to load cases dataframe, clean and process data."""

    cases_df: pd.DataFrame
    raw_cases_df: pd.DataFrame = None
    data_version: int = 2
    project_dir: str | Path = Path(__file__).parent.parent
    data_dir: Optional[str | Path] = None
    processed_case_lab_association_path: Optional[str | Path] = None
    case_lab_association_df: Optional[pd.DataFrame] = None
    admission_type_category = CategoricalDtype(
        categories=["Inpatient", "Observation", "Outpatient", "Unknown"], ordered=False
    )
    covid_case_interval_category: CategoricalDtype = CategoricalDtype(
        categories=[
            "0-1_Weeks",
            "1-2_Weeks",
            "2-3_Weeks",
            "3-4_Weeks",
            "4-5_Weeks",
            "5-6_Weeks",
            "6-7_Weeks",
            "7-8_Weeks",
            ">=8_Weeks",
            "Never",
        ],
        ordered=True,
    )
    covid_case_interval_category2: CategoricalDtype = CategoricalDtype(
        categories=[
            "0-2_Weeks",
            "2-4_Weeks",
            "4-6_Weeks",
            "6-8_Weeks",
            ">=8_Weeks",
            "Never",
        ],
        ordered=True,
    )

    def __post_init__(self) -> pd.DataFrame:
        "Called upon object instance creation."
        # Set Default Data Directories and Paths
        self.project_dir = Path(self.project_dir)
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / f"v{self.data_version}"
        else:
            self.data_dir = Path(self.data_dir)
        if self.processed_case_lab_association_path is None:
            self.processed_case_lab_association_path = (
                self.data_dir / "processed" / "case_associated_covid_labs.parquet"
            )

        # If path passed into cases_df argument, load dataframe from path
        if isinstance(self.cases_df, str | Path):
            df = read_pandas(self.cases_df)
            self.raw_cases_df = df.copy()

        # Format Cases Data
        self.cases_df = self.format_cases_df(df)

    def __call__(self) -> pd.DataFrame:
        return self.cases_df

    def format_cases_df(self, cases_df: pd.DataFrame) -> pd.DataFrame:
        """Clean cases dataframe.  This returns a new dataframe with different column values.
        Sets MPOG_Case_ID as index.

        Cases with missing values for Postop_LOS are set to 0 (assume no length of stay).

        Args:
            cases_df (pd.DataFrame): raw cases dataframe

        Returns:
            pd.DataFrame: transformed output dataframe
        """
        _df = cases_df.copy()
        _df.MPOG_Case_ID = _df.MPOG_Case_ID.str.upper()
        _df.MPOG_Patient_ID = _df.MPOG_Patient_ID.str.upper()
        # Set MPOG_Case_ID as primary index
        _df = _df.set_index("MPOG_Case_ID")

        ### Format Basic Case Info
        # Format Anesthesia Start into DateTime object
        anes_start = _df.AnesthesiaStart_Value.apply(
            lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
        )
        # Format Case Duration (min) into TimeDelta object
        anes_duration = _df.AnesthesiaDuration_Value.apply(
            lambda minutes: timedelta(minutes=minutes)
        )
        # Format PACU Duration into TimeDelta object
        pacu_duration = _df.PacuDuration_Value.apply(lambda minutes: timedelta(minutes=minutes))
        # Format Post-op Length of Stay into TimeDelta object
        postop_los_duration = _df.PostopLengthOfStayDays_Value.fillna(0).apply(
            lambda days: timedelta(days=days)
        )

        # Clean admission types
        def clean_admission_types(value: str) -> str:
            if value in ("Inpatient", "Admit", "Emergency"):
                return "Inpatient"
            elif value == "Outpatient":
                return "Outpatient"
            elif value == "23 hour observation":
                return "Observation"
            else:
                return "Unknown"

        admission_type = _df.AdmissionType_Value.apply(clean_admission_types).astype(
            self.admission_type_category
        )

        # Emergency Case?
        is_emergency = _df.EmergencyStatusClassification_Value.apply(
            lambda value: True if value == "Yes" else False
        )

        # Basic Case Info Dataframe
        case_df = pd.DataFrame(
            data={
                "MPOG_Patient_ID": _df.MPOG_Patient_ID,
                "AnesStart": anes_start,
                "AnesDuration": anes_duration,
                "Age": _df.AgeInYears_Value,
                "Race": _df.Race_Value,
                "Sex": _df.Sex_Value,
                "ASA": _df.AsaStatusClassification_Value,
                "IsEmergency": is_emergency,
                "BMI": _df.BodyMassIndex_Value,
                "Weight": _df.Weight_Value,
                "Height": _df.Height_Value,
                "Mortality30Day": _df.HospitalMortality30Day_Value,
                "MORT01": _df.MORT01_Result_Reason,
                "PACU_Duration": pacu_duration,
                "Postop_LOS": postop_los_duration,
                "AdmissionType": admission_type,
                "SurgeryRegion": _df.PrimaryAnesthesiaCPT_MPOGAnesCPTClass,
                "CardiacProcedureType": _df.ProcedureTypeCardiacAlt_Value,
                "IsCardiacProcedure": _df.ProcedureTypeCardiacAlt_Value == "No",
            },
            index=_df.index,
        )

        # Drop Invalid Anesthesia Case Durations (https://phenotypes.mpog.org/Anesthesia%20Duration)
        case_df = case_df.loc[case_df.AnesDuration > timedelta(minutes=0), :].loc[
            case_df.AnesDuration < timedelta(hours=36), :
        ]
        # Drop Invalid PACU Durations (https://phenotypes.mpog.org/PACU%20Duration)
        case_df = case_df.loc[case_df.PACU_Duration > timedelta(minutes=0), :].loc[
            case_df.PACU_Duration < timedelta(hours=20), :
        ]

        ### Elixhauser & MPOG Comorbidities
        elixhauser_comorbidities = [
            "ComorbidityElixhauserCardiacArrhythmias_Value",
            "ComorbidityElixhauserChronicPulmonaryDisease_Value",
            "ComorbidityElixhauserCongestiveHeartFailure_Value",
            "ComorbidityElixhauserDiabetesComplicated_Value",
            "ComorbidityElixhauserDiabetesUncomplicated_Value",
            "ComorbidityElixhauserHypertensionComplicated_Value",
            "ComorbidityElixhauserHypertensionUncomplicated_Value",
            "ComorbidityElixhauserLiverDisease_Value",
            "ComorbidityElixhauserMetastaticCancer_Value",
            "ComorbidityElixhauserObesity_Value",
            "ComorbidityElixhauserPeripheralVascularDisorders_Value",
            "ComorbidityElixhauserPulmonaryCirculationDisorders_Value",
            "ComorbidityElixhauserRenalFailure_Value",
            "ComorbidityElixhauserValvularDisease_Value",
        ]
        mpog_comorbidities = [
            "ComorbidityMpogCerebrovascularDisease_Value_Code",
            "ComorbidityMpogCoronaryArteryDisease_Value_Code",
        ]
        comorbidities = elixhauser_comorbidities + mpog_comorbidities
        # Drop Cases without ICD Codes documented for comorbidities, then convert to boolean
        comorbidities_df = (
            _df.loc[:, comorbidities]
            .applymap(self.booleanize_comorbidity)
            .astype(bool)
            .dropna(axis=0, how="any")
        )
        # Remove suffix _Value or _Value_Code
        comorbidities_df.columns = [col.split("_")[0] for col in comorbidities_df.columns]

        ### AHRQ & MPOG Complications
        ## Pulmonary Complications
        # Reference: https://phenotypes.mpog.org/AHRQ%20Complication%20-%20Pulmonary%20-%20All
        # Get all ICD codes for pulmonary complications
        pulm_ahrq_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_AHRQ_Diagnoses.replace(
            np.nan, None
        ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
        pulm_mpog_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_MPOG_Diagnoses.replace(
            np.nan, None
        ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
        pulm_icd_codes = (pulm_ahrq_diagnoses + pulm_mpog_diagnoses).apply(set).apply(list)

        def clean_pulm_complication_presence(value: str) -> str:
            if "Yes" in value:
                return "Yes"
            elif value == "No":
                return "No"
            elif "Unknown" in value:
                return "Unknown"
            else:
                raise ValueError(f"Unknown value {value} for clean_pulm_complication_presence")

        had_pulm_complication = _df.ComplicationAHRQPulmonaryAll_Value.apply(
            clean_pulm_complication_presence
        )

        ## Cardiac Complications

        def clean_cardiac_complication_presence(value: str) -> str:
            if value == 1:
                return "Yes"
            elif value == 0:
                return "No"
            elif value == -999:
                return "Unknown"
            else:
                raise ValueError(f"Unknown value {value} for clean_cardiac_complication_presence")

        had_cardiac_complication = _df.ComplicationAHRQCardiac_Value.apply(
            clean_cardiac_complication_presence
        )
        had_myocardial_infarction = _df.ComplicationAHRQMyocardialInfarction_Value.apply(
            lambda value: True if value == "YES" else False
        ).astype(bool)

        ## Renal Complications
        # Reference: https://phenotypes.mpog.org/MPOG%20Complication%20-%20Acute%20Kidney%20Injury%20(AKI)

        def clean_aki_complication_presence(value: str) -> str:
            if value in (1, 2, 3):
                return "AKI"
            elif value == 0:
                return "No AKI"
            elif value == -2:
                return "Pre-existing ESRD"
            elif value in (-1, -3, -999):
                return "Unknown"
            else:
                raise ValueError(f"Unknown value {value} for clean_aki_complication_presence")

        had_aki = _df.ComplicationMpogAcuteKidneyInjury_Value.apply(clean_aki_complication_presence)

        complications_df = pd.DataFrame(
            data={
                "HadAKIComplication": had_aki,
                "HadCardiacComplication": had_cardiac_complication,
                "HadMyocardialInfarctionComplication": had_myocardial_infarction,
                "HadPulmonaryComplication": had_pulm_complication,
                "PulmonaryComplicationICD": pulm_icd_codes,
            },
            index=_df.index,
        )

        ### Combine Basic Case Info, Comorbidities, Complications Dataframes
        common_indices = list(
            set.intersection(
                set(case_df.index), set(comorbidities_df.index), set(complications_df.index)
            )
        )
        case_df_subset = case_df.loc[common_indices, :]
        comorbidities_df_subset = comorbidities_df.loc[common_indices, :]
        complications_df_subset = complications_df.loc[common_indices, :]
        output_df = case_df_subset.join(comorbidities_df_subset).join(complications_df_subset)
        return output_df

    def booleanize_comorbidity(self, value: int) -> bool | float:
        """Elixhauser and MPOG Comorbidities in MPOG Dataset has 3 values in raw data:
            0 = No (No ICD-9 or ICD-10 codes that matches comorbidity)
            1 = Yes (Presence of ICD-9 or ICD-10 code that maches comorbidity)
            -999 = Unknown (No ICD-9 or ICD-10 codes recorded)

            References:
            - https://phenotypes.mpog.org/MPOG%20Comorbidity%20-%20Cerebrovascular%20Disease
            - https://phenotypes.mpog.org/Elixhauser%20Comorbidity%20-%20Cardiac%20Arrhythmias
        Args:
            value (int): comorbidity value

        Returns:
            bool: True (1), False (0) or np.nan (-999)
        """
        if value == 1:
            return True
        elif value == 0:
            return False
        elif value == -999:
            return np.nan
        else:
            raise ValueError("Cannot booleanize value {value} for comorbidity.")

    def associate_labs_to_cases(
        self,
        labs_df: Optional[pd.DataFrame] = None,
        processed_case_lab_association_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        if labs_df is None and processed_case_lab_association_path is None:
            raise ValueError(
                "Must supply either `labs_df` or `processed_case_lab_association_path` args."
            )
        if processed_case_lab_association_path is not None:
            self.processed_case_lab_association_path = processed_case_lab_association_path
        try:
            # Load cached result from disk, convert durations/intervals to timedelta
            case_lab_association_df = read_pandas(self.processed_case_lab_association_path)
            case_lab_association_df["LastPostitiveCovidInterval"] = case_lab_association_df[
                "LastPositiveCovidInterval"
            ].apply(pd.Timedelta)
            self.case_lab_association_df = case_lab_association_df.astype(
                {
                    "LastPositiveCovidInterval": "timedelta64[ns]",
                    "LabCaseIntervalCategory": self.covid_case_interval_category,
                    "LabCaseIntervalCategory2": self.covid_case_interval_category2,
                }
            )
        except FileNotFoundError:
            if not isinstance(labs_df, pd.DataFrame):
                raise ValueError("Must provide argument `labs_df`.")
            # Associate preop labs with cases for each patient, then combine into a single dataframe
            processed_labs_for_all_cases = []
            for cases_grp in tqdm(
                self.cases_df.groupby("MPOG_Patient_ID"),
                desc="Associating Labs to Cases",
            ):
                # Get dataframe of cases for each patient
                mpog_patient_id, cases = cases_grp
                # Get labs only for patient
                labs = labs_df.loc[labs_df.MPOG_Patient_ID == mpog_patient_id]
                # For each patient's cases, get last pre-op covid lab test
                processed_labs_for_cases = cases.apply(
                    lambda row: self.last_covid_lab_before_case(
                        labs_df=labs, mpog_patient_id=mpog_patient_id, case_start=row.AnesStart
                    ),
                    axis=1,
                ).apply(pd.Series)
                # Accumulate for all patients and cases
                processed_labs_for_all_cases += [processed_labs_for_cases]

            self.case_lab_association_df = pd.concat(processed_labs_for_all_cases).astype(
                {
                    "LastPositiveCovidInterval": "timedelta64[ns]",
                    "LabCaseIntervalCategory": self.covid_case_interval_category,
                    "LabCaseIntervalCategory2": self.covid_case_interval_category2,
                }
            )
            # Cache result on disk, convert timedelta to durations/intervals
            case_lab_association_df = self.case_lab_association_df.copy()
            case_lab_association_df["LastPositiveCovidInterval"] = case_lab_association_df[
                "LastPositiveCovidInterval"
            ].apply(lambda x: x.isoformat())
            case_lab_association_df.to_parquet(self.processed_case_lab_association_path)

        # Join labs to cases_df
        self.cases_df = self.cases_df.join(self.case_lab_association_df)
        return self.cases_df

    def categorize_duration(self, duration: timedelta) -> str | None:
        """Creates buckets of duration similar to how COVIDSurg 2021 paper did it.
        'Timing of surgery following SARS-CoV-2 infection: an international prospective
        cohort study' (https://pubmed.ncbi.nlm.nih.gov/33690889/)

        Create 1-week intervals.

        Args:
            duration (timedelta): duration of time used to create category/bin.

        Returns:
            str: String description of category of covid case interval.
                For time durations of 0 and negative, None is returned.
        """
        if duration in (pd.NaT, np.nan, None) or duration < timedelta(seconds=0):
            return "Never"
        elif duration >= timedelta(seconds=0) and duration < timedelta(weeks=1):
            return "0-1_Weeks"
        elif duration >= timedelta(weeks=1) and duration < timedelta(weeks=2):
            return "1-2_Weeks"
        elif duration >= timedelta(weeks=2) and duration < timedelta(weeks=3):
            return "2-3_Weeks"
        elif duration >= timedelta(weeks=3) and duration < timedelta(weeks=4):
            return "3-4_Weeks"
        elif duration >= timedelta(weeks=4) and duration < timedelta(weeks=5):
            return "4-5_Weeks"
        elif duration >= timedelta(weeks=5) and duration < timedelta(weeks=6):
            return "5-6_Weeks"
        elif duration >= timedelta(weeks=6) and duration < timedelta(weeks=7):
            return "6-7_Weeks"
        elif duration >= timedelta(weeks=7) and duration < timedelta(weeks=8):
            return "7-8_Weeks"
        else:
            return ">=8_Weeks"

    def categorize_duration2(self, duration: timedelta) -> str | None:
        """Creates buckets of duration similar to how COVIDSurg 2021 paper did it.
        'Timing of surgery following SARS-CoV-2 infection: an international prospective
        cohort study' (https://pubmed.ncbi.nlm.nih.gov/33690889/)

        Creates 2-week intervals.

        Args:
            duration (timedelta): duration of time used to create category/bin.

        Returns:
            str: String description of category of covid case interval.
                For time durations of 0 and negative, None is returned.
        """
        if duration in (pd.NaT, np.nan, None) or duration < timedelta(seconds=0):
            return "Never"
        elif duration >= timedelta(seconds=0) and duration < timedelta(weeks=2):
            return "0-2_Weeks"
        elif duration >= timedelta(weeks=2) and duration < timedelta(weeks=4):
            return "2-4_Weeks"
        elif duration >= timedelta(weeks=4) and duration < timedelta(weeks=6):
            return "4-6_Weeks"
        elif duration >= timedelta(weeks=6) and duration < timedelta(weeks=8):
            return "6-8_Weeks"
        else:
            return ">=8_Weeks"

    def last_covid_lab_before_case(
        self, labs_df: pd.DataFrame, mpog_patient_id: str, case_start: pd.Timestamp | datetime
    ) -> dict[str, Any]:
        """Get most recent COVID lab test before case start datetime.  Lab values occuring
        after case_start are ignored.

        Args:
            labs_df (pd.DataFrame): cleaned labs dataframe
            mpog_patient_id (str): MPOG unique patient identifier
            case_start (pd.Timestamp | datetime): anesthesia start time

        Returns:
            dict[str, Any]: dictionary of most recent covid result as well as whether patient
                had positive covid result lab test in the stratified time intervals.
        """
        labs = labs_df.loc[labs_df.MPOG_Patient_ID == mpog_patient_id, :].sort_values(
            by="DateTime", ascending=True
        )
        # Create Timedelta objects for duration interval between lab and case
        lab_case_interval = case_start - labs.DateTime
        lab_case_interval_category = lab_case_interval.apply(self.categorize_duration).astype(
            self.covid_case_interval_category
        )
        lab_case_interval_category2 = lab_case_interval.apply(self.categorize_duration2).astype(
            self.covid_case_interval_category2
        )
        # Get Only Pre-op Labs Prior to Case
        labs["IsPreopLab"] = labs.DateTime < case_start
        preop_labs = labs.loc[labs.IsPreopLab, :]
        # Get Only Positive Pre-op Labs Prior to Case
        positive_preop_labs = preop_labs.loc[preop_labs.Result == "Positive"]
        has_positive_preop_covid_test = not positive_preop_labs.empty
        if has_positive_preop_covid_test:
            last_positive_preop_covid_test = positive_preop_labs.tail(n=1)
            last_positive_preop_covid_labuuid = last_positive_preop_covid_test.index.item()
            last_positive_preop_covid_datetime = last_positive_preop_covid_test.DateTime.item()
            last_positive_preop_covid_interval = lab_case_interval.loc[
                last_positive_preop_covid_labuuid
            ]
            last_positive_preop_covid_interval_category = lab_case_interval_category.loc[
                last_positive_preop_covid_labuuid
            ]
            last_positive_preop_covid_interval_category2 = lab_case_interval_category2.loc[
                last_positive_preop_covid_labuuid
            ]
        else:
            last_positive_preop_covid_labuuid = None
            last_positive_preop_covid_datetime = pd.NaT
            last_positive_preop_covid_interval = pd.NaT
            last_positive_preop_covid_interval_category = "Never"
            last_positive_preop_covid_interval_category2 = "Never"

        return {
            "EverCovidPositive": any(preop_labs.Result == "Positive"),
            "HasPositivePreopCovidTest": has_positive_preop_covid_test,
            "LastPositiveCovidLabUUID": last_positive_preop_covid_labuuid,
            "LastPositiveCovidDateTime": last_positive_preop_covid_datetime,
            "LastPositiveCovidInterval": last_positive_preop_covid_interval,
            "LabCaseIntervalCategory": last_positive_preop_covid_interval_category,
            "LabCaseIntervalCategory2": last_positive_preop_covid_interval_category2,
        }
