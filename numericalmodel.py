import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
red_channel = pd.read_csv('/Users/riddhishsaravanan/BioData/DNAM/red.csv')
green_channel = pd.read_csv('/Users/riddhishsaravanan/BioData/DNAM/outputs.csv')
methylation_final = pd.merge(red_channel, green_channel, on='NBeads', suffixes=('_red', '_green'))
methylation_final.replace("--", np.nan, inplace=True)
genomic_data = pd.read_csv('/Users/riddhishsaravanan/BioData/Copy Number Variation/f66f2a1c-8a63-4d92-a963-4ee8fbd78b0d/CNV.tsv', sep='\t')
epigenetic_data = pd.read_csv('/Users/riddhishsaravanan/BioData/Transcriptome Profiling/81d92351-c619-4585-9281-de33eaaabba4/Transcriptome.tsv', sep='\t')
genetic_final = pd.merge(genomic_data, epigenetic_data, on='gene_id')
genetic_final.replace("--", np.nan, inplace=True)
clinical_data = pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/clinical.tsv', sep='\t')
exposure_data = pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/exposure.tsv', sep='\t')
clinical_final = pd.merge(clinical_data, exposure_data, on='case_id')
clinical_final.replace("--", np.nan, inplace=True)

maf_file_path = '/Users/riddhishsaravanan/BioData/SNV.maf'
snv_final = pd.read_csv(maf_file_path, sep='\t', comment='#')
snv_final.replace("--", np.nan, inplace=True)

# Preprocess function with checks
def preprocess_data(df, feature_columns):
    features = df[feature_columns]
    print(f'Shape before preprocessing: {features.shape}')  # Print shape for debugging
    features.fillna(features.mean(), inplace=True)  # Fill NaN values with mean
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f'Shape after preprocessing: {features_scaled.shape}')  # Print shape after scaling
    return features_scaled


clinical_cols = ['Tumor_Grade_x', 'age_at_index', 'age_is_obfuscated', 'days_to_birth', 'days_to_death', 'year_of_birth', 'year_of_death', 'age_at_diagnosis', 'days_to_best_overall_response', 'days_to_diagnosis', 'days_to_last_follow_up', 'days_to_last_known_disease_status', 'days_to_recurrence', 'year_of_diagnosis', 'age_at_last_exposure', 'age_at_onset', 'alcohol_days_per_week', 'alcohol_drinks_per_day', 'alcohol_frequency', 'cigarettes_per_day', 'exposure_duration_years', 'years_smoked', 'bmi', 'height', 'weight']
clinical_final[clinical_cols] = clinical_final[clinical_cols].apply(pd.to_numeric, errors='coerce')

# Define feature lists
clinical_features = clinical_cols

# Convert columns in genetic_final to numeric
genetic_cols = ['copy_number', 'min_copy_number', 'max_copy_number']
genetic_final[genetic_cols] = genetic_final[genetic_cols].apply(pd.to_numeric, errors='coerce')

# Define feature list
genetic_epigenetic_features = genetic_cols

# Convert columns in methylation_final to numeric
methylation_cols = ['SD_red', 'SD_green', 'NBeads', 'Mean_red', 'Mean_green']
methylation_final[methylation_cols] = methylation_final[methylation_cols].apply(pd.to_numeric, errors='coerce')

# Define feature list
methylation_features = methylation_cols

# Convert columns in snv_final to numeric
snv_cols = ['Entrez_Gene_Id', 'Start_Position', 'End_Position', 't_depth', 't_ref_count', 't_alt_count', 'n_depth']
snv_final[snv_cols] = snv_final[snv_cols].apply(pd.to_numeric, errors='coerce')

# Define feature list
snv_features = snv_cols

# Preprocess each dataframe
X_clinical = preprocess_data(clinical_final, clinical_features)
X_genetic_epigenetic = preprocess_data(genetic_final, genetic_epigenetic_features)
X_methylation = preprocess_data(methylation_final, methylation_features)
X_snv = preprocess_data(snv_final, snv_features)

# Concatenate all preprocessed features
X = np.concatenate([X_clinical, X_genetic_epigenetic, X_methylation, X_snv], axis=1)

# Assuming target variable is the same for all dataframes
target = clinical_final['Tumor_Grade_x']  # Replace with actual target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
