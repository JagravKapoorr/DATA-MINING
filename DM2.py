from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer,Binarizer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import resample
import pandas as pd 
iris = load_iris()
X, y = iris.data, iris.target
# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
df_standardized = pd.DataFrame(X_standardized, columns=iris.feature_names)
df_standardized.head()
# Normalization
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
df_normalized = pd.DataFrame(X_normalized, columns=iris.feature_names)
df_normalized.head()
# Discretization
kbins_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_discretized = kbins_discretizer.fit_transform(X)
df_discretized = pd.DataFrame(X_discretized, columns=iris.feature_names)
df_discretized.head()
# Aggregation (KMeans clustering)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
X_clustered = kmeans.labels_
df_clustered = pd.DataFrame({'Cluster': X_clustered})
df_clustered.head()
df_clustered.tail()
# Feature Selection (SelectKBest using chi-squared test)
selector = SelectKBest(chi2, k=2)
X_selected = selector.fit_transform(X, y)
df_selected = pd.DataFrame(X_selected, columns=['Feature1', 'Feature2'])
df_selected.head()
# Binarization
binarizer = Binarizer(threshold=3.0)  # Example threshold
X_binarized = binarizer.fit_transform(X)
df_binarized = pd.DataFrame(X_binarized, columns=iris.feature_names)
df_binarized.head()
# Sampling (random sampling with replacement)
X_resampled, y_resampled = resample(X, y, replace=True, random_state=42, n_samples=50)  # Example size
df_resampled = pd.DataFrame(X_resampled, columns=iris.feature_names)
df_resampled.head()