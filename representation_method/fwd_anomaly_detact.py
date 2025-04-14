import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --- Load and Prepare Data ---
csv_path = Path(__file__).parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv'
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

# Filter for a specific recording session if desired
df = df[df['RECORDING_SESSION_LABEL'] == 2]

# Define target: Here 0 = normal, 1 = non conscious (e.g., NODULE_HIT)
df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', 1, 0)

# Create a group identifier based on RECORDING_SESSION_LABEL and TRIAL_INDEX to ensure groups stay together
from sklearn.model_selection import train_test_split

# --- Completely Random Data Splitting ---
# Instead of grouping by session and trial, we perform a completely random split.
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

# --- Feature Preprocessing ---
# Define the features you want to use for anomaly detection.
features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_COMPONENT_DURATION']

# Convert features to numeric and handle missing values.
for feature in features:
    train_df[feature] = pd.to_numeric(train_df[feature], errors='coerce').fillna(0)
    test_df[feature] = pd.to_numeric(test_df[feature], errors='coerce').fillna(0)

# Standardize the features.
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])

y_train = train_df['target'].values
y_test = test_df['target'].values

# --- Anomaly Detection using IsolationForest ---
# Train the anomaly detection model using only the normal samples (target == 0).
X_train_normal = X_train[y_train == 0]
iso_forest = IsolationForest(contamination='auto', random_state=42)
iso_forest.fit(X_train_normal)

# Predict on the test set.
# Note: IsolationForest returns 1 for normal and -1 for anomalies.
predictions = iso_forest.predict(X_test)
# Map predictions to a binary anomaly label: 0 = normal, 1 = anomaly.
predicted_anomalies = np.where(predictions == -1, 1, 0)

# Append predictions to the test DataFrame.
test_df['predicted_anomaly'] = predicted_anomalies

# --- Evaluate Anomaly Detection ---
# In our setup, target==1 (non conscious gaze) are the anomalies.
cm = confusion_matrix(y_test, predicted_anomalies)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predicted_anomalies))

# --- Optional: Plot the Distribution of Anomaly Scores ---
scores_test = iso_forest.decision_function(X_test)
plt.figure(figsize=(8, 6))
plt.hist(scores_test, bins=30, color='blue', alpha=0.7)
plt.title("Distribution of Anomaly Scores on Test Data")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# --- Load and Prepare Data ---
csv_path = Path(__file__).parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv'
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

# Filter for a specific recording session if desired
df = df[df['RECORDING_SESSION_LABEL'] == 1]

df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', 1, 0)

from sklearn.model_selection import train_test_split

# --- Completely Random Data Splitting ---
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

# --- Feature Preprocessing ---
# Define the features you want to use
features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_COMPONENT_DURATION']

# Convert features to numeric and fill missing values with 0
for feature in features:
    train_df[feature] = pd.to_numeric(train_df[feature], errors='coerce').fillna(0)
    test_df[feature] = pd.to_numeric(test_df[feature], errors='coerce').fillna(0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])
y_train = train_df['target'].values
y_test = test_df['target'].values

# --- GMM Classifier ---
gmm_normal = GaussianMixture(n_components=1, random_state=42)
gmm_non_conscious = GaussianMixture(n_components=1, random_state=42)

X_train_normal = X_train[y_train == 0]
X_train_non_conscious = X_train[y_train == 1]

gmm_normal.fit(X_train_normal)
gmm_non_conscious.fit(X_train_non_conscious)

# Compute prior probabilities from the training set
prior_normal = X_train_normal.shape[0] / X_train.shape[0]
prior_non_conscious = X_train_non_conscious.shape[0] / X_train.shape[0]

# For each test sample, compute the log likelihood under each model
log_likelihood_normal = gmm_normal.score_samples(X_test)
log_likelihood_non_conscious = gmm_non_conscious.score_samples(X_test)

# Incorporate the class priors (in log-space) to compute log posteriors
log_post_normal = np.log(prior_normal) + log_likelihood_normal
log_post_non_conscious = np.log(prior_non_conscious) + log_likelihood_non_conscious

# Predict the class with the higher log posterior probability
y_pred_gmm = np.where(log_post_non_conscious > log_post_normal, 1, 0)

# --- Evaluation ---
cm = confusion_matrix(y_test, y_pred_gmm)
print("Confusion Matrix for GMM Classifier:")
print(cm)

print("\nClassification Report for GMM Classifier:")
print(classification_report(y_test, y_pred_gmm))

# --- Optional: Visualize the Log Posterior Distributions ---
plt.figure(figsize=(8, 6))
plt.hist(log_post_normal, bins=30, alpha=0.7, label='Normal class log-posterior')
plt.hist(log_post_non_conscious, bins=30, alpha=0.7, label='Non-conscious class log-posterior')
plt.title("Log Posterior Distributions for GMM Classifier")
plt.xlabel("Log Posterior")
plt.ylabel("Frequency")
plt.legend()
plt.show()

csv_path = Path(__file__).parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv'
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

# Filter for a specific recording session if desired
df = df[df['RECORDING_SESSION_LABEL'] == 1]

# Define target: 0 = normal, 1 = non conscious (e.g., NODULE_HIT)
df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', 1, 0)

from sklearn.model_selection import train_test_split

# --- Completely Random Data Splitting ---
# Instead of grouping by session and trial, we perform a completely random split.
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

# --- Preprocessing CURRENT_FIX_DURATION ---
# Convert CURRENT_FIX_DURATION to numeric and fill missing values with 0
train_df['CURRENT_FIX_DURATION'] = pd.to_numeric(train_df['CURRENT_FIX_DURATION'], errors='coerce').fillna(0)
test_df['CURRENT_FIX_DURATION'] = pd.to_numeric(test_df['CURRENT_FIX_DURATION'], errors='coerce').fillna(0)

# --- Compute Class Statistics ---
# Calculate the mean for each class on the training set
mean_normal = train_df.loc[train_df['target'] == 0, 'CURRENT_FIX_DURATION'].mean()
mean_non_conscious = train_df.loc[train_df['target'] == 1, 'CURRENT_FIX_DURATION'].mean()

# Set a threshold as the average of the two means
threshold = (mean_normal + mean_non_conscious) -1000

print(f"Mean CURRENT_FIX_DURATION - Normal: {mean_normal:.3f}, Non-Conscious: {mean_non_conscious:.3f}")
print(f"Threshold used for classification: {threshold:.3f}")

# --- Simple Threshold-Based Classifier ---
# Determine which group has the higher mean
# If non-conscious has a higher mean, then a value > threshold will be classified as non-conscious (1)
# Otherwise, if non-conscious has a lower mean, a value < threshold will be classified as non-conscious (1)
if mean_non_conscious > mean_normal:
    test_df['predicted'] = np.where(test_df['CURRENT_FIX_DURATION'] > threshold, 1, 0)
else:
    test_df['predicted'] = np.where(test_df['CURRENT_FIX_DURATION'] < threshold, 1, 0)

# --- Evaluation ---
cm = confusion_matrix(test_df['target'], test_df['predicted'])
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(test_df['target'], test_df['predicted']))

# --- Optional: Visualize the Distribution ---
plt.figure(figsize=(8,6))
plt.hist(train_df.loc[train_df['target'] == 0, 'CURRENT_FIX_DURATION'], bins=30, alpha=0.7, label='Normal')
plt.hist(train_df.loc[train_df['target'] == 1, 'CURRENT_FIX_DURATION'], bins=30, alpha=0.7, label='Non-Conscious')
plt.axvline(threshold, color='black', linestyle='dashed', label=f'Threshold: {threshold:.2f}')
plt.title("Distribution of CURRENT_FIX_DURATION in Training Data")
plt.xlabel("CURRENT_FIX_DURATION")
plt.ylabel("Frequency")
plt.legend()
plt.show()