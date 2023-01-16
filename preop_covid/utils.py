import hashlib
from pathlib import Path

import pandas as pd


def read_pandas(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("read_pandas cannot read file extension.")


def create_uuid(data: str, output_format: str = "T-SQL") -> str:
    """Creates unique UUID using BLAKE2b algorithm.

    Args:
        data (str): input data used to generate UUID
        output_format (str): Output format.
            `raw` results in raw 32-char digest being returned as UUID.
            `T-SQL` results in 36-char UUID string (32 hex values, 4 dashes)
                delimited in the same style as `uniqueidentifier` in T-SQL databases.

    Returns:
        Formatted UUID.
    """
    data = data.encode("UTF-8")  # type: ignore
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()  # type: ignore
    if output_format == "raw":
        return digest
    elif output_format == "T-SQL":
        uuid = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:]}"
        return uuid
    else:
        raise ValueError(f"Unknown argument {output_format} specified for `return_format`.")
