from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from utils import read_pandas


@dataclass
class CaseData:
    """Data object to load cases dataframe, clean and process data."""

    cases_df: str | Path | pd.DataFrame
    raw_cases_df: pd.DataFrame = None
    admission_type_category = CategoricalDtype(
        categories=["Inpatient", "Observation", "Outpatient", "Unknown"], ordered=False
    )
    covid_case_interval_category = CategoricalDtype(
        categories=["0-2_weeks", "3-4_weeks", "5-6_weeks", ">=7_weeks"], ordered=True
    )

    def __post_init__(self) -> pd.DataFrame:
        if isinstance(self.cases_df, str | Path):
            df = read_pandas(self.cases_df)
            self.raw_cases_df = copy.deepcopy(df)
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
