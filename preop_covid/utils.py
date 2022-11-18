import hashlib


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
