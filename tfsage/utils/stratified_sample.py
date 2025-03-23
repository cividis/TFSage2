import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    n_samples: int | None = None,
    p_positive: float | None = None,
    random_state: int | None = None,
):
    """
    Perform a stratified sampling of a DataFrame based on specified criteria.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to sample.
            It must include a "positive" column to identify positive samples.
        n_samples (int | None, optional): The total number of samples to draw.
            If specified, the function will sample this many rows while
            maintaining the original class proportions or the specified
            proportion of positive samples. Defaults to None.
        p_positive (float | None, optional): The desired proportion of positive
            samples in the output. If specified, the function will adjust the
            number of positive and negative samples accordingly. Defaults to None.
        random_state (int | None, optional): A random seed for reproducibility
            of the sampling process. Defaults to None.

    Returns:
        pd.DataFrame: A stratified sample of the input DataFrame, sorted by
        "chrom", "start", and "end" columns, with the index reset.
    """
    df_positive = df.query("positive")
    df_negative = df.query("~positive")

    if n_samples is not None:
        n_samples = min(n_samples, len(df))

    if p_positive is not None:
        # Limit the number of samples based on the class proportions
        n_samples = min(
            n_samples,
            int(len(df_positive) / p_positive),
            int(len(df_negative) / (1 - p_positive)),
        )
        n_positive = min(int(n_samples * p_positive), len(df_positive))
        n_negative = n_samples - n_positive

    if p_positive is not None:
        df_positive = df_positive.sample(n=n_positive, random_state=random_state)
        df_negative = df_negative.sample(n=n_negative, random_state=random_state)
        df = pd.concat([df_positive, df_negative])
    elif n_samples is not None:
        # Case: only sample size specified – sample with original class proportions
        df = df.sample(n=n_samples, random_state=random_state)

    # Sort by chrom, start, end
    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    return df
