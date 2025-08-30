import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv('loan_data.csv')

# Encode previous_loan_defaults_on_file
data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'Yes': 0, 'No': 1})

# One-hot encode other categoricals
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent']
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated parameter
encoded_cats = encoder.fit_transform(data[categorical_cols])

# Get feature names
encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols)

# Drop original categoricals and concat
data = data.drop(columns=categorical_cols)
data = pd.concat([data, encoded_df], axis=1)

# Save preprocessed data
data.to_pickle('preprocessed_data.pkl')
print("Preprocessed data saved as preprocessed_data.pkl")