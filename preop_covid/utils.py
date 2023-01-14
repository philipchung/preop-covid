import hashlib


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


def clean_covid_result_value(value: str) -> str:
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
