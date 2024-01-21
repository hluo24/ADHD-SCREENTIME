!pip uninstall scikit-learn --yes
!pip uninstall imblearn --yes
!pip install scikit-learn==1.2.2
!pip install imblearn

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import converter

file_dicts = [
    {
        'sas7bdat_file': 'nsch_2022e_topical.sas7bdat',
        'export_file': 'Blank-CSV-Template.csv',
    },
    {
        'sas7bdat_file': 'nsch_2022e_topical.sas7bdat',
        'export_file': 'Blank-CSV-Template.csv',
    },
]
converter.batch_to_csv(file_dicts)
df = pd.read_csv("Blank-CSV-Template.csv")

newdf = df.drop(['K2Q31B', 'K2Q31D', 'ADDTREAT', 'K2Q31C', 'SC_K2Q10', 'SC_K2Q11', 'SC_K2Q12', 'SC_K2Q22', 'SC_K2Q23'], axis=1)
train = newdf.copy()
train.loc[:, 'K2Q31A'] = train['K2Q31A'].fillna(2)
train = train.fillna(0)
train = pd.get_dummies(train, columns=['STRATUM', 'FORMTYPE', 'INQ_RESSEG', 'INQ_EDU', 'INQ_EMPLOY', 'INQ_INCOME', 'INQ_HOME'])

X = train.drop('K2Q31A', axis=1)

# Assuming X is your feature matrix
imputer = SimpleImputer(strategy='mean', fill_value=0)  # You can choose 'mean', 'median', 'most_frequent', etc.
X_imputed = imputer.fit_transform(X)

y = train['K2Q31A']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)


oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

# Fit and apply the oversampler to the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Now, you can use the resampled data to train your model
rf = RandomForestClassifier()
rf.fit(X_train_resampled, y_train_resampled)

y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Select features based on feature importances
sfm = SelectFromModel(rf, threshold=0.02)  # Adjust threshold as needed
sfm.fit(X_train, y_train)

# Transform the feature matrix
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Print selected feature indices
selected_features = sfm.get_support(indices=True)
print("Selected Features:", selected_features)

conf_matrix = confusion_matrix(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1.0)  # Specify pos_label
recall = recall_score(y_test, y_pred, pos_label=1.0)  # Specify pos_label
f1 = f1_score(y_test, y_pred, pos_label=1.0)  # Specify pos_label

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

plt.bar(range(len(selected_features)), selected_features)
plt.xlabel('Feature Number')
plt.ylabel('Importance')
plt.show()

selected_feature_names = X.columns[selected_features]

# Print or use the selected feature names
print("Selected Feature Names:", selected_feature_names)
