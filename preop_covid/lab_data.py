from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from pandas.api.types import CategoricalDtype
from utils import create_uuid, read_pandas


@dataclass
class LabData:
    """Data object to load labs dataframe, clean and process data."""

    labs_df: str | Path | pd.DataFrame
    raw_labs_df: pd.DataFrame = None
    data_version: int = 2
    project_dir: str | Path = Path(__file__).parent.parent
    data_dir: Optional[str | Path] = None
    covid_lab_result_category: CategoricalDtype = CategoricalDtype(
        categories=["Negative", "Positive", "Unknown"], ordered=False
    )
    covid_lab_values_map: dict = field(default_factory=dict)
    covid_lab_values_map_path: str | Path = Path(__file__).parent / "covid_lab_values_map.yml"

    def __post_init__(self) -> None:
        "Called upon object instance creation."
        # Set Default Data Directories and Paths
        self.project_dir = Path(self.project_dir)
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / f"v{self.data_version}"
        else:
            self.data_dir = Path(self.data_dir)

        # Load mapping of raw to cleaned covid lab values
        if not self.covid_lab_values_map:
            try:
                self.covid_lab_values_map = yaml.safe_load(
                    Path(self.covid_lab_values_map_path).read_text()
                )
            except Exception:
                raise ValueError(
                    "Must provide either `covid_lab_values_map` or `covid_lab_values_map_path`."
                )

        # If path passed into labs_df argument, load dataframe from path
        if isinstance(self.labs_df, str | Path):
            df = read_pandas(self.labs_df)
            # Normalize headers from space-separated words to underscore-separated
            df.columns = [col_title.replace(" ", "_") for col_title in df.columns]
            self.raw_labs_df = df.copy()

        # Format Lab Data
        self.labs_df = self.format_labs_df(df)

    def __call__(self) -> pd.DataFrame:
        return self.labs_df

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
        # If there are any duplicated UUIDS, the row entry has the same value for all 3 of these.
        _df = labs_df.copy()
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
        if self.data_version == 1:
            lab_datetime = _df.AIMS_Lab_Observation_DT.apply(
                lambda s: datetime.strptime(s, r"%Y-%m-%d %H:%M:%S")
            )
        elif self.data_version in (1.1, 2.0):
            lab_datetime = _df.AIMS_Lab_Observation_DT.apply(
                lambda s: datetime.strptime(s, r"%m/%d/%y %H:%M")
            )
        else:
            raise ValueError("Unknown String DateTime Format in `AIMS_Lab_Observation_DT`.")
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
        """Converts covid result value field into Categorical value.

        Args:
            value (str): string value for HMTopic field

        Returns:
            str: "Positive", "Negative", "Unknown"
        """
        value = value.lower().strip()
        if value in [x.lower() for x in self.covid_lab_values_map["positive_values"]]:
            return "Positive"
        elif value in [x.lower() for x in self.covid_lab_values_map["negative_values"]]:
            return "Negative"
        elif value in [x.lower() for x in self.covid_lab_values_map["unknown_values"]]:
            return "Unknown"
        else:
            raise ValueError(
                f"Unknown value {value} encountered that is not handled by "
                "clean_covid_result_value() logic."
            )
