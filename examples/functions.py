import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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

def calculate_npmi(X, Y = None):
  """
  Calculate Normalized Pointwise Mutual Information (NPMI) between two matrices X and Y.
  The NPMI is defined as: NPMI(x,y) = log(P(x,y) / (P(x) * P(y))) / -log(P(x,y))

  Args:
      X: CSR Matrix, first matrix of feature activations (shape (F, N)), where F is the number of features and N is the number of samples
      Y: CSR Matrix, second matrix of feature activations (shape (F, N))
  """
  is_symmetric = Y is None
  Y = Y or X

  assert X.dtype == int or X.dtype == np.int32 or X.dtype == np.int64, f"For NPMI, X must be int, not {X.dtype}"
  assert Y.dtype == int or Y.dtype == np.int32 or Y.dtype == np.int64, f"For NPMI, Y must be int, not {Y.dtype}"
  n_features_X, n_samples = X.shape
  n_features_Y = Y.shape[0]

  # Calculate P_XY
  cooc_sparse = X @ Y.T  # Cross co-occurrence matrix
  cooc_sparse.eliminate_zeros()
  rows, cols = cooc_sparse.nonzero()
  cooc_data = cooc_sparse.data

  feature_counts_X = np.asarray(X.sum(axis=1)).flatten()
  feature_counts_Y = np.asarray(Y.sum(axis=1)).flatten()
  eps = 1e-10
  P_x = (feature_counts_X + eps) / n_samples
  P_y = (feature_counts_Y + eps) / n_samples
  P_xy_vals = (cooc_data + eps) / n_samples
  P_x_vals = P_x[rows]
  P_y_vals = P_y[cols]

  pmi_vals = np.log(P_xy_vals / (P_x_vals * P_y_vals))
  log_P_xy_vals = np.log(P_xy_vals)
  npmi_vals = np.where(np.abs(log_P_xy_vals) < eps, 1.0, pmi_vals / (-log_P_xy_vals))

  npmi_vals = np.nan_to_num(npmi_vals, nan=0.0, posinf=1.0, neginf=-1.0)
  npmi_sparse = csr_matrix((npmi_vals, (rows, cols)), shape=(n_features_X, n_features_Y))
  if is_symmetric:
    npmi_sparse.setdiag(1.0)
  return npmi_sparse