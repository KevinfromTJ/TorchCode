"""PCA Reconstruction-Error Anomaly Detection — OPPO 2026 8.8 (Problem 2).

ML problem (numpy / scikit-learn), function-mode.
Given a reference/train matrix of normal samples and a test matrix (both may
contain None/NaN missing values), detect anomalies via PCA reconstruction error:
  1. Fill missing values with the per-column mean of the TRAIN matrix (train
     stats only) for both train and test.
  2. StandardScaler fit on train, transform both.
  3. PCA(n_components=0.95, svd_solver='full') fit on the scaled train; project
     and inverse_transform both to get reconstructions.
  4. Reconstruction error = sum of squared per-feature differences.
  5. Threshold = np.percentile(train_errors, 95). Label 1 (anomaly) iff a test
     sample's error is STRICTLY greater than the threshold, else 0.
Return the list of labels in test order.
"""

TASK = {
    "title": "PCA Anomaly Detection",
    "difficulty": "Medium",
    "function_name": "detect_anomaly",
    "hint": (
        "np.array(..., dtype=float) turns None into nan. means = np.nanmean(train, axis=0); "
        "fill nan in train AND test with means[col]. StandardScaler().fit(train) then transform "
        "both. PCA(n_components=0.95, svd_solver='full').fit(train_scaled); reconstruct via "
        "pca.inverse_transform(pca.transform(X)). Errors are np.sum((X - X_rec)**2, axis=1). "
        "threshold = np.percentile(train_err, 95); return (test_err > threshold).astype(int)."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
train = [[1, 2], [2, 4.1], [3, 5.9], [4, 8.1], [5, 10]]
test = [[2.5, 5.0], [2.5, 10.0], [3.0, None]]
assert {fn}(train, test) == [0, 1, 0], {fn}(train, test)
""",
        },
        {
            "name": "Output format: one 0/1 label per test row, in order",
            "code": """
train = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.1], [4.0, 3.9], [5.0, 5.0], [6.0, 6.0]]
test = [[1.5, 1.5], [100.0, -100.0], [2.0, 2.0]]
labels = {fn}(train, test)
assert isinstance(labels, list) and len(labels) == 3
assert all(l in (0, 1) for l in labels), labels
assert labels[1] == 1, labels            # extreme outlier
""",
        },
        {
            "name": "Clear outlier flagged, inlier not",
            "code": """
import random
random.seed(0)
# strongly correlated 2D normal data along y = x
train = [[x + random.uniform(-0.05, 0.05), x + random.uniform(-0.05, 0.05)] for x in range(2, 20)]
test = [[10.0, 10.0], [10.0, -10.0]]   # 2nd breaks the correlation
labels = {fn}(train, test)
assert labels[1] == 1, labels
assert labels[0] == 0, labels
""",
        },
        {
            "name": "Matches reference implementation on random data",
            "code": """
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def ref(train, test):
    X = np.array(train, dtype=float); T = np.array(test, dtype=float)
    means = np.nanmean(X, axis=0)
    r, c = np.where(np.isnan(X)); X[r, c] = means[c]
    r, c = np.where(np.isnan(T)); T[r, c] = means[c]
    sc = StandardScaler(); Xs = sc.fit_transform(X); Ts = sc.transform(T)
    pca = PCA(n_components=0.95, svd_solver='full'); pca.fit(Xs)
    Xr = pca.inverse_transform(pca.transform(Xs))
    Tr = pca.inverse_transform(pca.transform(Ts))
    etr = np.sum((Xs - Xr) ** 2, axis=1); ete = np.sum((Ts - Tr) ** 2, axis=1)
    thr = np.percentile(etr, 95)
    return (ete > thr).astype(int).tolist()

random.seed(11)
for _ in range(50):
    n = random.randint(4, 15); m = random.randint(2, 12); d = random.randint(2, 6)
    train = [[random.uniform(-5, 5) for _ in range(d)] for _ in range(n)]
    test = [[random.uniform(-8, 8) for _ in range(d)] for _ in range(m)]
    # sprinkle a few missing values
    if random.random() < 0.5:
        train[0][0] = None
    if random.random() < 0.5:
        test[0][-1] = None
    assert {fn}([r[:] for r in train], [r[:] for r in test]) == ref(train, test)
""",
        },
    ],
}
