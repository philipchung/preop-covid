from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from utils import create_uuid, parallel_process, read_pandas


@dataclass
class PreopData:
    """Data object to load structured data from preop note (e.g. ROS),
    clean and process data."""

    preop_df: str | Path | pd.DataFrame
    processed_preop_df_path: Optional[str | Path] = None
    workflow_df: pd.DataFrame = None
    ros_df: pd.DataFrame = None
    problems_df: pd.DataFrame = None
    data_version: int = 2
    project_dir: str | Path = Path(__file__).parent.parent
    data_dir: Optional[str | Path] = None

    def __post_init__(self) -> pd.DataFrame:
        "Called upon object instance creation."
        # Set Default Data Directories and Paths
        self.project_dir = Path(self.project_dir)
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / f"v{self.data_version}"
        else:
            self.data_dir = Path(self.data_dir)

        # If path passed into preop_df argument, load dataframe from path
        if self.processed_preop_df_path is None:
            self.processed_preop_df_path = (
                self.data_dir / "processed" / "preop_smartdataelements_cleaned.parquet"
            )

        # Load Preop Data
        self.preop_df = self.format_preop_df(preop_df=self.preop_df)

        # Only "WORKFLOW" SmartDataElement fields
        self.workflow_df = self.preop_df.loc[
            self.preop_df.SmartDataElement.str.contains("WORKFLOW -")
        ].copy()
        # Only "WORKFLOW - ROS" SmartDataElement fields
        self.ros_df = self.workflow_df.loc[
            self.workflow_df.SmartDataElement.str.contains("- ROS -")
        ].copy()
        # Only "DIAGNOSES/PROBLEMS" SmartDataElement fields
        self.problems_df = self.preop_df.loc[
            self.preop_df.SmartDataElement.str.contains("DIAGNOSES/PROBLEMS -")
        ].copy()

    def format_preop_df(self, preop_df: pd.DataFrame | str | Path) -> pd.DataFrame:
        """Clean Preop SmartDataElements dataframe.  This returns a new dataframe
        with different column values.
        Generates a UUID for each SmartDataElement field, which is unique as long
        as we have a unique MPOG Case ID, MPOG Patient ID, SmartDataElement.

        Args:
            preop_df (pd.DataFrame | str | Path): Preop SmartDataElements dataframe.
                If this is not a filepath, then we assume this is the dataframe itself
                and this is a noop.

        Returns:
            pd.DataFrame: Same Dataframe with SmartDataElementUUID
        """
        # Load Preop Data (from cache if possible), Combine Lines
        # Note: the cached processed preop_df drops columns "AIMS_Case_ID" & "Line"
        if isinstance(self.preop_df, str | Path):
            try:
                df = read_pandas(self.processed_preop_df_path)
                self.preop_df = df
            except FileNotFoundError:
                df = read_pandas(preop_df)
                # Combine Lines (see caveat in function definition)
                df = self.combine_lines(df)
                # Create UUID for each SmartDataElement & Use as Index
                sde_uuid = self.create_uuid(df)
                df = df.assign(SmartDataElementUUID=sde_uuid).set_index("SmartDataElementUUID")
                # Cache Processed Dataframe
                df.to_parquet(self.processed_preop_df_path)
                self.preop_df = df.copy()
        return df

    def create_uuid(self, df: pd.DataFrame) -> pd.Series:
        """Create UUID for each SmartDataElement based on MPOG Case ID, MPOG Patient ID,
        SmartDataElement name. If there are any duplicated UUIDs, the row entry has the
        same value for all 3 of these

        Args:
            df (pd.DataFrame): dataframe that must contain columns "MPOG_Case_ID",
                "MPOG_Patient_ID", "SmartDataElement".

        Returns:
            pd.Series: series of UUIDs generated for each row of the input df.
        """
        result = parallel_process(
            iterable=(t._asdict() for t in df.itertuples(index=False)),
            function=create_uuid_for_row_tuple,
            desc="Creating SmartDataElement UUIDs",
        )
        return pd.Series(result, name="SmartDataElementUUID").astype("string")

    def combine_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Epic SmartDataElement comment values that cannot fit within Clarity's
        limit length get split across multiple "Line".  This method joins all
        lines for same SmartDataElement back together.

        Args:
            df (pd.DataFrame): table of SmartDataElements for
                all MPOG_Case_ID and all MPOG_Patient_ID

        Returns:
            pd.DataFrame: table of SmartDataElements for
                all MPOG_Case_ID and all MPOG_Patient_ID with "Line" concatenated
                together.
        """
        groups = df.groupby(["MPOG_Case_ID", "MPOG_Patient_ID", "SmartDataElement"])[
            "SmartElemValue"
        ]

        # Important Caveat Here:
        # It is possible to have multiple pre-anesthesia notes generated for
        # each MPOG_Case_ID.  To uniquely identify a SmartDataElement in a note, we should
        # actually require a unique tuple of MPOG_Case_ID, Note ID, SmartDataElement. Since
        # we do not have the Note IDs, if multiple Pre-Anesthesia notes were written for
        # a patient for a MPOG_Case_ID, then we have all the SmartDataElements from all the
        # Pre-Anesthesia notes.  When we combine the Lines from the SmartDataElements,
        # we are doing this indiscriminantly across all possible Pre-Anesthesia notes for
        # the case, and all of them get combined into the same text string. The effect is
        # taking the union of all SmartDataElements across all Pre-Anesthesia notes for
        # the MPOG_Case_ID. For this project, it does not matter--we will only look at
        # whether someone checked "+" or wrote a comment into a SmartDataElement,
        # we will not look to see exactly what the comment was.

        # If "+", "WORKFLOW - ROS - XXX COMMENTS" = 1
        # If text comments, "WORKFLOW - ROS - XXX COMMENTS" = text comment
        # If "-", "WORKFLOW - ROS - NEG/NEGATIVE XXX ROS" = 1

        result = parallel_process(
            iterable=groups,
            function=concatenate_lines_for_groups,
            use_args=True,
            desc="Combining Lines for Each Field",
        )

        return pd.DataFrame(
            result,
            columns=[
                "MPOG_Case_ID",
                "MPOG_Patient_ID",
                "SmartDataElement",
                "SmartElemValue",
            ],
        ).astype(
            {
                "MPOG_Case_ID": "string",
                "MPOG_Patient_ID": "string",
                "SmartDataElement": "string",
                "SmartElemValue": "string",
            },
        )


def concatenate_lines_for_groups(indices: tuple, text_list: list[str] | pd.Series) -> tuple:
    concatenated_text_field = concatenate_lines(text_list)
    return (*indices, concatenated_text_field)


def concatenate_lines(text_list: list[str] | pd.Series) -> str:
    """Concatenates a list of text or a pd.Series with text into a single string.

    Args:
        text_list (list[str] | pd.Series): list of text.  pd.Series may sometimes only
            have 1 element, in which .item() method must be used to access the contents.
            If pd.Series has 2+ elements, .tolist() is used to access the contents.

    Raises:
        ValueError: for invalid input types

    Returns:
        str: string text from list or pd.Series concatenated into a single string
    """
    if isinstance(text_list, pd.Series):
        if len(text_list) == 1:
            return text_list.item()
        else:
            return "".join(text_list.tolist())
    elif isinstance(text_list, list):
        return "".join(text_list)
    else:
        raise ValueError(f"Unknown input type {type(text_list)} for argument `text_list`.")


def create_uuid_for_row_tuple(row_dict: dict[str, str]) -> str:
    return create_uuid(
        str(row_dict["MPOG_Patient_ID"])
        + str(row_dict["MPOG_Case_ID"])
        + str(row_dict["SmartDataElement"])
    )
