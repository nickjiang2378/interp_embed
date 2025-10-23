import numpy as np
import pandas as pd

def diff_features(
    ds1,
    ds2,
    metric="absolute",
    min_coverage=0.0,
    max_coverage=1.0
):
    """
    Calculate the differences in feature frequency (i.e. how often a feature is activated) between two datasets.

    :param other: Another DatasetFeatureActivations instance
    :param metric: Metric to use for calculating the difference ('absolute', 'relative')
    :param min_coverage: Minimum percentage of samples that must have a non-zero activation to consider a feature
    :param max_coverage: Maximum percentage of samples that must have a non-zero activation to consider a feature
    :return: Pandas DataFrame with feature labels and their frequency differences
    """
    # Get per-feature activations (frequency of nonzero activation for each feature)
    fa1 = ds1.latents("binarize")  # shape: (D1, F)
    fa2 = ds2.latents("binarize")  # shape: (D2, F)

    freq1 = np.sum(fa1, axis=0) / fa1.shape[0]
    freq2 = np.sum(fa2, axis=0) / fa2.shape[0]

    # Mask features outside coverage bounds
    min_freq = np.minimum(freq1, freq2)
    max_freq = np.maximum(freq1, freq2)
    mask = (min_freq < min_coverage) | (max_freq > max_coverage)
    freq1_masked = freq1.copy()
    freq2_masked = freq2.copy()
    freq1_masked[mask] = -1
    freq2_masked[mask] = -1

    # Calculate the difference according to metric
    if metric == "absolute":
        diff = freq1_masked - freq2_masked
    elif metric == "relative":
        denom = np.maximum(freq1_masked, freq2_masked).copy()
        denom[denom == 0] = 1  # avoid division by zero
        diff = (freq1_masked / denom) - (freq2_masked / denom)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Get feature labels
    feature_labels_dict = ds1.feature_labels()
    num_features = freq1.shape[0]
    feature_labels = [feature_labels_dict.get(i, "") for i in range(num_features)]

    df = pd.DataFrame({
        "feature": feature_labels,
        "feature_id": np.arange(num_features),
        ds1.id: freq1_masked,
        ds2.id: freq2_masked,
        "frequency_difference": diff,
    })

    df = df.sort_values(by=["frequency_difference", f"{ds1.id}"], ascending=False, ignore_index=True)
    return df

def find_correlations(ds1, ds2):
  pass