import pandas as pd
red_channel = pd.read_csv('/Users/riddhishsaravanan/BioData/DNAM/red.csv')
green_channel = pd.read_csv('/Users/riddhishsaravanan/BioData/DNAM/outputs.csv')
methylation_final = pd.merge(red_channel, green_channel, on='NBeads', suffixes=('_red', '_green'))

# Load the TSV files
genomic_data = pd.read_csv('/Users/riddhishsaravanan/BioData/Copy Number Variation/f66f2a1c-8a63-4d92-a963-4ee8fbd78b0d/CNV.tsv', sep='\t')
epigenetic_data = pd.read_csv('/Users/riddhishsaravanan/BioData/Transcriptome Profiling/81d92351-c619-4585-9281-de33eaaabba4/Transcriptome.tsv', sep='\t')
genetic_final = pd.merge(genomic_data, epigenetic_data, on='gene_id')
#Load Clinical Data
clinical_data=pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/clinical.tsv',sep='\t')
exposure_data=pd.read_csv('/Users/riddhishsaravanan/BioData/clinical/exposure.tsv',sep='\t')
clinical_final=pd.merge(clinical_data,exposure_data,on='case_id')

print(methylation_final.columns)
