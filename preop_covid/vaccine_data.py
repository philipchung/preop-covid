import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from pandas.api.types import CategoricalDtype
from tqdm.auto import tqdm
from utils import create_uuid, parallel_process, read_pandas


@dataclass
class VaccineData:
    """Data object to load health maintenance dataframe (which contains vaccines),
    clean and process data."""

    vaccines_df: str | Path | pd.DataFrame
    raw_vaccines_df: pd.DataFrame = None
    data_version: int = 2
    project_dir: str | Path = Path(__file__).parent.parent
    data_dir: Optional[str | Path] = None
    cleaned_vaccines_df_path: Optional[str | Path] = None
    health_maintenance_map: pd.DataFrame = field(default_factory=dict)
    health_maintenance_map_path: str | Path = (
        Path(__file__).parent / "health_maintenance_values_map.yml"
    )
    covid_vaccine_codes_df: pd.DataFrame = field(default_factory=dict)
    covid_vaccine_codes_path: Optional[str | Path] = None
    vaccine_category: CategoricalDtype = CategoricalDtype(
        categories=["COVID-19", "Influenza", "Other"], ordered=False
    )
    covid_vaccine_kind: CategoricalDtype = CategoricalDtype(
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

    def __post_init__(self) -> pd.DataFrame:
        "Called upon object instance creation."
        # Set Default Data Directories and Paths
        self.project_dir = Path(self.project_dir)
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / f"v{self.data_version}"
        else:
            self.data_dir = Path(self.data_dir)
        if self.cleaned_vaccines_df_path is None:
            self.cleaned_vaccines_df_path = (
                self.data_dir / "processed" / "cleaned_vaccines_df_path.parquet"
            )

        # Load mapping of health maintenance categories used to sort vaccine category
        if not self.health_maintenance_map:
            try:
                self.health_maintenance_map = yaml.safe_load(
                    Path(self.health_maintenance_map_path).read_text()
                )
            except Exception:
                raise ValueError(
                    "Must provide either `health_maintenance_map` or `health_maintenance_map_path`."
                )
        # Load covid vaccine codes & crosswalk info from CDC
        if not self.covid_vaccine_codes_df:
            try:
                self.covid_vaccine_codes_path = self.data_dir / "mappings" / "vaccine_codes.csv"
                self.covid_vaccine_codes_df = pd.read_csv(self.covid_vaccine_codes_path)
            except Exception:
                raise ValueError(
                    "Must provide either `covid_vaccine_codes_df` or `covid_vaccine_codes_path`."
                )

        # If path passed into vaccine_df argument, load dataframe from path
        if isinstance(self.vaccines_df, str | Path):
            df = read_pandas(self.vaccines_df)
            self.raw_vaccines_df = df.copy()

        # Format Vaccines Data
        self.vaccines_df = self.format_vaccines_df(df)

    def format_vaccines_df(self, vaccines_df: pd.DataFrame) -> pd.DataFrame:
        """Clean health maintenance dataframe and keep only vaccine information.
        This returns a new dataframe with different column values.
        Generates a UUID for each vaccine administration, which is unique as long as we have
        unique MPOG Patient ID, Health Maintenance Topic, and Health Maintenance DateTime.
        Then we uniquify vaccine entries based on this UUID and use it as an index.

        Args:
            vaccines_df (pd.DataFrame): raw health maintenance (vaccines) dataframe

        Returns:
            pd.DataFrame: transformed output dataframe
        """
        _df = vaccines_df.copy()
        # Drop Incompleted Vaccines
        _df = _df.loc[_df.HMType == "Done"]
        # Clean Vaccine Type
        _df["VaccineType"] = _df.HMTopic.apply(self.clean_hm_topic_value).astype(
            self.vaccine_category
        )
        # Keep Only COVID-19 and Influenza Vaccines
        _df = _df.loc[_df.VaccineType != "Other"]
        # Create UUID for each Vaccine Administration based on MPOG Patient ID,
        # Health Maintenance Topic, Health Maintenance DateTime.  If there are any
        # duplicated UUIDs, the row entry has the same value for all 3 of these.
        vaccine_uuid = _df.apply(
            lambda row: create_uuid(
                str(row.MPOG_Patient_ID) + str(row.HMTopic) + str(row.HM_HX_DATE)
            ),
            axis=1,
        )
        _df["VaccineUUID"] = vaccine_uuid
        # Remove Case Info
        _df = _df.loc[
            :, ["VaccineUUID", "MPOG_Patient_ID", "HM_HX_DATE", "COMMENTS", "VaccineType"]
        ]

        # Try to load cached data if it exists
        try:
            vaccines_df = pd.read_parquet(self.cleaned_vaccines_df_path)
            # Check to see if same VaccineUUIDs
            assert set(vaccines_df.index.tolist()) == set(_df["VaccineUUID"].tolist())
            # If pass, then we can use the cached result
            self.vaccines_df = vaccines_df
            self.flu_vaccines_df = vaccines_df.loc[vaccines_df.VaccineType == "Influenza"]
            self.covid_vaccines_df = vaccines_df.loc[vaccines_df.VaccineType == "COVID-19"]
            return self.vaccines_df
        except FileNotFoundError:
            # No cached file, so we need to cleanup the vaccine data

            # Unfortunately we can't uniquify based on VaccineUUID yet at this point.
            # There are some vaccine COMMENTS for flu vaccine that are different even though
            # the administration date is the same.  We can't drop COMMENTS yet because we
            # need them to further categorize COVID-19 vaccine.

            # Isolate Flu Vaccines
            _flu_df = _df.loc[_df.VaccineType == "Influenza"].copy()
            _flu_df = _flu_df.assign(VaccineKind="Influenza")
            # Uniquify based on VaccineUUID, then use it as index
            _flu_df = (
                _flu_df.drop_duplicates(subset="VaccineUUID")
                .drop(columns="COMMENTS")
                .set_index("VaccineUUID")
            )
            self.flu_vaccines_df = _flu_df

            # Isolate & Further Clean/Format COVID-19
            _covid_df = _df.loc[_df.VaccineType == "COVID-19"].copy()
            # Use regex to split string
            tqdm.pandas(desc="Parsing Covid Vaccine Comments")
            self.covid_matches = pd.DataFrame(
                _covid_df.COMMENTS.progress_apply(self.parse_covid_vaccine_comments).tolist()
            )
            # Clean trailing characters that are sometimes present
            self.covid_matches.Vaccine = self.covid_matches.Vaccine.str.rstrip(
                "\x11\x110"
            ).str.strip()
            # Categorize Vaccine Kind (e.g. Moderna, Pfizer, etc.)
            fn = partial(
                categorize_covid_vaccine_kind, covid_vaccine_codes_df=self.covid_vaccine_codes_df
            )
            result = parallel_process(
                iterable=self.covid_matches.Vaccine,
                function=fn,
                desc="Categorizing Covid Vaccine Kind",
            )
            _covid_df = _covid_df.assign(VaccineKind=result)
            # Uniquify based on VaccineUUID, then use it as index
            _covid_df = (
                _covid_df.drop_duplicates(subset="VaccineUUID")
                .drop(columns="COMMENTS")
                .set_index("VaccineUUID")
            )
            self.covid_vaccines_df = _covid_df

            # Combined Vaccine Table
            self.vaccines_df = pd.concat([_flu_df, _covid_df], axis=0)
            # Cache Result to Disk
            self.vaccines_df.to_parquet(self.cleaned_vaccines_df_path)
        return self.vaccines_df

    def clean_hm_topic_value(self, value: str) -> str:
        """Converts health maintenance topic field into Categorical value.

        Args:
            value (str): string value for HMTopic field

        Returns:
            str: "COVID-19", "Influenza", "Other"
        """
        value = value.lower().strip()
        if value in [x.lower() for x in self.health_maintenance_map["covid-19_vaccine"]]:
            return "COVID-19"
        elif value in [x.lower() for x in self.health_maintenance_map["influenza_vaccine"]]:
            return "Influenza"
        elif value in [x.lower() for x in self.health_maintenance_map["other"]]:
            return "Other"
        else:
            raise ValueError(
                f"Unknown value {value} encountered that is not handled by "
                "clean_hm_topic_value() logic."
            )

    def parse_covid_vaccine_comments(self, text: str) -> Optional[dict[str, str]]:
        """Use regular expression to parse COVID-19 vaccine COMMENTS field.

        Args:
            text (str): COMMENTS field from health maintenance table

        Returns:
            str: Dictionary of extracted groups with keys
                ["AdminInfo", "LIMID", "LPLID", "Vaccine"]
        """
        pattern = re.compile(
            "(?P<AdminInfo>.*)"  # Administration Type
            ":\s"  # Colon separator
            "(?P<LIMID>LIMI[dD].*?)?"  # LIM ID
            "\^?"  # ^ separator
            "(?P<LPLID>LPLId.*?)?"  # LPL ID
            "\^\\x11"  # ^\x11 separator
            "(?P<Vaccine>.*)"  # Vaccine Name & Details
        )
        grp_names = ["AdminInfo", "LIMID", "LPLID", "Vaccine"]
        match = re.search(pattern, text)
        if match is not None:
            return {name: match.group(name) for name in grp_names}
        else:
            return None


def categorize_covid_vaccine_kind(text: str, covid_vaccine_codes_df: pd.DataFrame) -> str:
    """Categorizes cleaned covid vaccine string description into major vaccine kinds.

    Args:
        text (str): input text string describing covid vaccine
        covid_vaccine_codes_df (pd.DataFrame): table of covid vaccine codes from CDC

    Returns:
        str: one of the following categories ["Moderna", "Pfizer", "Janssen",
            "Novavax", "Sanofi", "AstraZeneca", "Unspecified mRNA Vaccine", "Other"]
    """
    text_lower = text.lower()

    moderna_cvx_short_description = covid_vaccine_codes_df.loc[
        covid_vaccine_codes_df.Manufacturer == "Moderna US, Inc."
    ]["CVX Short Description"].tolist()

    pfizer_cvx_short_description = covid_vaccine_codes_df.loc[
        covid_vaccine_codes_df.Manufacturer == "Pfizer-BioNTech"
    ]["CVX Short Description"].tolist()

    janssen_cvx_short_description = covid_vaccine_codes_df.loc[
        covid_vaccine_codes_df.Manufacturer == "Janssen Products, LP"
    ]["CVX Short Description"].tolist()

    novavax_cvx_short_description = covid_vaccine_codes_df.loc[
        covid_vaccine_codes_df.Manufacturer == "Novavax, Inc."
    ]["CVX Short Description"].tolist()

    sanofi_cvx_short_description = covid_vaccine_codes_df.loc[
        covid_vaccine_codes_df.Manufacturer == "Sanofi Pasteur"
    ]["CVX Short Description"].tolist()

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
