from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import CategoricalDtype
from utils import create_uuid, read_pandas


@dataclass
class LabData:
    labs_df: str | Path | pd.DataFrame
    covid_lab_result_category: CategoricalDtype = CategoricalDtype(
        categories=["Negative", "Positive", "Unknown"], ordered=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.labs_df, str | Path):
            self.labs_df = read_pandas(self.labs_df)

    def format_labs_df(self, labs_df: pd.DataFrame) -> pd.DataFrame:
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
        lab_result = _df.AIMS_Value_Text.apply(self.clean_covid_result_value).astype(
            self.covid_lab_result_category
        )
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

    def clean_covid_result_value(self, value: str) -> str:
        """Converts result values into Categorical value Positive, Negative, Unknown."""
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
            "Follow-up testing required. Refer to other SARS-CoV-2 Qualitative PCR result on specimen with similar collection date and time.",  # noqa:E501
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
