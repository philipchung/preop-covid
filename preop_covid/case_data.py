from __future__ import annotations

import copy
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
    project_dir: str | Path = Path(__file__).parent.parent
    processed_case_lab_association_path: Optional[str | Path] = None
    case_lab_association_df: Optional[pd.DataFrame] = None
    admission_type_category = CategoricalDtype(
        categories=["Inpatient", "Observation", "Outpatient", "Unknown"], ordered=False
    )
    covid_case_interval_category = CategoricalDtype(
        categories=["0-2_weeks", "3-4_weeks", "5-6_weeks", ">=7_weeks"], ordered=True
    )

    def __post_init__(self) -> pd.DataFrame:
        "Called upon object instance creation."
        # If path passed into cases_df argument, load dataframe from path
        if isinstance(self.cases_df, str | Path):
            df = read_pandas(self.cases_df)
        self.raw_cases_df = copy.deepcopy(df)
        self.cases_df = self.format_cases_df(df)

        # Set Default Data Directories and Paths
        self.project_dir = Path(self.project_dir)
        self.data_dir = self.project_dir / "data" / "v1"
        if self.processed_case_lab_association_path is None:
            self.processed_case_lab_association_path = (
                self.data_dir / "processed" / "case_associated_covid_labs.parquet"
            )

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
        _df = copy.deepcopy(cases_df)

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

        # Get all ICD codes for pulmonary complications
        pulm_ahrq_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_AHRQ_Diagnoses.replace(
            np.nan, None
        ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
        pulm_mpog_diagnoses = _df.ComplicationAHRQPulmonaryAll_Triggering_MPOG_Diagnoses.replace(
            np.nan, None
        ).apply(lambda input_str: [s.strip() for s in input_str.split(";")] if input_str else [])
        pulm_icd_codes = (pulm_ahrq_diagnoses + pulm_mpog_diagnoses).apply(set).apply(list)

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

        output_df = pd.DataFrame(
            {
                "MPOG_Case_ID": _df.MPOG_Case_ID,
                "MPOG_Patient_ID": _df.MPOG_Patient_ID,
                "AnesStart": anes_start,
                "AnesDuration": anes_duration,
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
                "AdmissionType": admission_type,
                "SurgeryRegion": _df.PrimaryAnesthesiaCPT_MPOGAnesCPTClass,
                "CardiacProcedureType": _df.ProcedureTypeCardiacAlt_Value,
                "IsCardiacProcedure": _df.ProcedureTypeCardiacAlt_Value == "No",
                "PulmonaryComplication": _df.ComplicationAHRQPulmonaryAll_Value,
                "PulmonaryComplicationICD": pulm_icd_codes,
            }
        ).set_index("MPOG_Case_ID")

        # Drop Invalid Anesthesia Case Durations (https://phenotypes.mpog.org/Anesthesia%20Duration)
        output_df = output_df.loc[output_df.AnesDuration > timedelta(minutes=0), :].loc[
            output_df.AnesDuration < timedelta(hours=36), :
        ]
        # Drop Invalid PACU Durations (https://phenotypes.mpog.org/PACU%20Duration)
        output_df = output_df.loc[output_df.PACU_Duration > timedelta(minutes=0), :].loc[
            output_df.PACU_Duration < timedelta(hours=20), :
        ]

        # Drop Unknown Admission Types
        output_df = output_df.loc[output_df.AdmissionType != "Unknown"]
        return output_df

    def associate_labs_to_cases(
        self,
        labs_df: Optional[pd.DataFrame] = None,
        processed_case_lab_association_path: Optional[str | Path] = None,
        update_cases_df: bool = True,
    ) -> pd.DataFrame:
        if labs_df is None and processed_case_lab_association_path is None:
            raise ValueError(
                "Must supply either `labs_df` or `processed_case_lab_association_path` args."
            )
        if processed_case_lab_association_path:
            self.processed_case_lab_association_path = processed_case_lab_association_path
        try:
            # Load cached result from disk
            self.case_lab_association_df = read_pandas(
                self.processed_case_lab_association_path
            ).astype({"LastPositivePreopCovidIntervalCategory": self.covid_case_interval_category})
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
                {"LastPositivePreopCovidIntervalCategory": self.covid_case_interval_category}
            )
            # Cache result on disk
            self.case_lab_association_df.to_parquet(self.processed_case_lab_association_path)

        if update_cases_df:
            self.cases_df = self.cases_df.join(self.case_lab_association_df)
        return self.case_lab_association_df

    def categorize_duration(self, duration: timedelta) -> str | None:
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
                had positive covid result lab test in the stratified time intervals:
                0-2 weeks, 3-4 weeks, 5-6 weeks, >7 weeks.
        """
        labs = labs_df.loc[labs_df.MPOG_Patient_ID == mpog_patient_id, :].sort_values(
            by="DateTime", ascending=True
        )
        # Create Timedelta objects for duration interval between lab and case
        labs["LabCaseInterval"] = case_start - labs.DateTime
        labs["LabCaseIntervalCategory"] = (
            labs["LabCaseInterval"]
            .apply(self.categorize_duration)
            .astype(self.covid_case_interval_category)
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
            last_positive_preop_covid_interval = (
                last_positive_preop_covid_test.LabCaseInterval.item()
            )
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
