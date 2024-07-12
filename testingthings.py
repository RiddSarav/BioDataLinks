import pandas as pd
row_name='alcohol_frequency'
clinical_data = pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/clinical.tsv', sep='\t')
exposure_data = pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/exposure.tsv', sep='\t')
clinical_final = pd.merge(clinical_data, exposure_data, on='case_id')
clinical_final[row_name] = pd.to_numeric(clinical_final[row_name], errors='coerce')
print(clinical_final)
