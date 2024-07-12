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

# Select feature columns for each dataframe
clinical_features = ['Tumor_Grade_x', 'case_id', 'case_submitter_id_x', 'project_id_x', 'age_at_index', 'age_is_obfuscated', 'cause_of_death', 'cause_of_death_source', 'country_of_birth', 'country_of_residence_at_enrollment', 'days_to_birth', 'days_to_death', 'education_level', 'ethnicity', 'gender', 'marital_status', 'occupation_duration_years_x', 'premature_at_birth', 'race', 'vital_status', 'weeks_gestation_at_birth', 'year_of_birth', 'year_of_death', 'adrenal_hormone', 'age_at_diagnosis', 'ajcc_clinical_m', 'ajcc_clinical_n', 'ajcc_clinical_stage', 'ajcc_clinical_t', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t', 'ajcc_staging_system_edition', 'ann_arbor_b_symptoms', 'ann_arbor_b_symptoms_described', 'ann_arbor_clinical_stage', 'ann_arbor_extranodal_involvement', 'ann_arbor_pathologic_stage', 'best_overall_response', 'burkitt_lymphoma_clinical_variant', 'cancer_detection_method', 'child_pugh_classification', 'clark_level', 'classification_of_tumor', 'cog_liver_stage', 'cog_neuroblastoma_risk_group', 'cog_renal_stage', 'cog_rhabdomyosarcoma_risk_group', 'contiguous_organ_invaded', 'days_to_best_overall_response', 'days_to_diagnosis', 'days_to_last_follow_up', 'days_to_last_known_disease_status', 'days_to_recurrence', 'diagnosis_is_primary_disease', 'double_expressor_lymphoma', 'double_hit_lymphoma', 'eln_risk_classification', 'enneking_msts_grade', 'enneking_msts_metastasis', 'enneking_msts_stage', 'enneking_msts_tumor_site', 'ensat_clinical_m', 'ensat_pathologic_n', 'ensat_pathologic_stage', 'ensat_pathologic_t', 'esophageal_columnar_dysplasia_degree', 'esophageal_columnar_metaplasia_present', 'fab_morphology_code', 'figo_stage', 'figo_staging_edition_year', 'first_symptom_longest_duration', 'first_symptom_prior_to_diagnosis', 'gastric_esophageal_junction_involvement', 'gleason_grade_group', 'gleason_grade_tertiary', 'gleason_patterns_percent', 'gleason_score', 'goblet_cells_columnar_mucosa_present', 'icd_10_code', 'igcccg_stage', 'inpc_grade', 'inpc_histologic_group', 'inrg_stage', 'inss_stage', 'international_prognostic_index', 'irs_group', 'irs_stage', 'ishak_fibrosis_score', 'iss_stage', 'last_known_disease_status', 'laterality', 'margin_distance', 'margins_involved_site', 'masaoka_stage', 'max_tumor_bulk_site', 'medulloblastoma_molecular_classification', 'melanoma_known_primary', 'metastasis_at_diagnosis', 'metastasis_at_diagnosis_site', 'method_of_diagnosis', 'micropapillary_features', 'mitosis_karyorrhexis_index', 'mitotic_count', 'morphology', 'ovarian_specimen_status', 'ovarian_surface_involvement', 'papillary_renal_cell_type', 'pediatric_kidney_staging', 'peritoneal_fluid_cytological_status', 'pregnant_at_diagnosis', 'primary_diagnosis', 'primary_disease', 'primary_gleason_grade', 'prior_malignancy', 'prior_treatment', 'progression_or_recurrence', 'residual_disease', 'satellite_nodule_present', 'secondary_gleason_grade', 'site_of_resection_or_biopsy', 'sites_of_involvement', 'sites_of_involvement_count', 'supratentorial_localization', 'synchronous_malignancy', 'tissue_or_organ_of_origin', 'tumor_burden', 'tumor_confined_to_organ_of_origin', 'tumor_depth', 'tumor_focality', 'tumor_grade_category', 'tumor_regression_grade', 'uicc_clinical_m', 'uicc_clinical_n', 'uicc_clinical_stage', 'uicc_clinical_t', 'uicc_pathologic_m', 'uicc_pathologic_n', 'uicc_pathologic_stage', 'uicc_pathologic_t', 'uicc_staging_system_edition', 'ulceration_indicator', 'weiss_assessment_findings', 'weiss_assessment_score', 'who_cns_grade', 'who_nte_grade', 'wilms_tumor_histologic_subtype', 'year_of_diagnosis', 'anaplasia_present', 'anaplasia_present_type', 'breslow_thickness', 'circumferential_resection_margin', 'greatest_tumor_dimension', 'gross_tumor_weight', 'largest_extrapelvic_peritoneal_focus', 'lymph_node_involved_site', 'lymph_nodes_positive', 'lymph_nodes_tested', 'lymphatic_invasion_present', 'non_nodal_regional_disease', 'non_nodal_tumor_deposits', 'percent_tumor_invasion', 'perineural_invasion_present', 'peripancreatic_lymph_nodes_positive', 'peripancreatic_lymph_nodes_tested', 'transglottic_extension', 'tumor_largest_dimension_diameter', 'tumor_stage', 'vascular_invasion_present', 'vascular_invasion_type', 'chemo_concurrent_to_radiation', 'clinical_trial_indicator', 'course_number', 'days_to_treatment_end', 'days_to_treatment_start', 'drug_category', 'embolic_agent', 'initial_disease_status', 'lesions_treated_number', 'number_of_cycles', 'number_of_fractions', 'prescribed_dose', 'protocol_identifier', 'radiosensitizing_agent', 'reason_treatment_ended', 'reason_treatment_not_given', 'regimen_or_line_of_therapy', 'residual_disease.1', 'route_of_administration', 'therapeutic_agents', 'therapeutic_level_achieved', 'therapeutic_levels_achieved', 'therapeutic_target_level', 'timepoint_category', 'treatment_anatomic_site', 'treatment_anatomic_sites', 'treatment_arm', 'treatment_dose', 'treatment_dose_max', 'treatment_dose_units', 'treatment_duration', 'treatment_effect', 'treatment_effect_indicator', 'treatment_frequency', 'treatment_intent_type', 'treatment_or_therapy', 'treatment_outcome', 'treatment_outcome_duration', 'treatment_type ', 'Tumor_Grade_y', 'case_submitter_id_y', 'project_id_y', 'age_at_last_exposure', 'age_at_onset', 'alcohol_days_per_week', 'alcohol_drinks_per_day', 'alcohol_frequency', 'alcohol_history', 'alcohol_intensity', 'alcohol_type', 'asbestos_exposure', 'asbestos_exposure_type', 'chemical_exposure_type', 'cigarettes_per_day', 'coal_dust_exposure', 'environmental_tobacco_smoke_exposure', 'exposure_duration', 'exposure_duration_hrs_per_day', 'exposure_duration_years', 'exposure_source', 'exposure_type', 'occupation_duration_years_y', 'occupation_type', 'pack_years_smoked', 'parent_with_radiation_exposure', 'radon_exposure', 'respirable_crystalline_silica_exposure', 'secondhand_smoke_as_child', 'smoking_frequency', 'time_between_waking_and_first_smoke', 'tobacco_smoking_onset_year', 'tobacco_smoking_quit_year', 'tobacco_smoking_status', 'type_of_smoke_exposure', 'type_of_tobacco_used', 'use_per_day', 'years_smoked', 'bmi', 'height', 'marijuana_use_per_week', 'smokeless_tobacco_quit_age', 'tobacco_use_per_day', 'weight']
genetic_epigenetic_features = ['gene_id', 'gene_name_x', 'chromosome', 'start', 'end', 'copy_number',
       'min_copy_number', 'max_copy_number', 'gene_name_y', 'gene_type',
       'unstranded', 'stranded_first', 'stranded_second', 'tpm_unstranded',
       'fpkm_unstranded', 'fpkm_uq_unstranded']
methylation_features = ['Mean', 'SD_red', 'SD_green', 'NBeads', 'Mean_red', 'Mean_green']
snv_features = ['Hugo_Symbol', 'Entrez_Gene_Id', 'Center', 'NCBI_Build', 'Chromosome', 'Start_Position', 'End_Position', 'Strand', 'Variant_Classification', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'dbSNP_RS', 'dbSNP_Val_Status', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2', 'Tumor_Validation_Allele1', 'Tumor_Validation_Allele2', 'Match_Norm_Validation_Allele1', 'Match_Norm_Validation_Allele2', 'Verification_Status', 'Validation_Status', 'Mutation_Status', 'Sequencing_Phase', 'Sequence_Source', 'Validation_Method', 'Score', 'BAM_File', 'Sequencer', 'Tumor_Sample_UUID', 'Matched_Norm_Sample_UUID', 'HGVSc', 'HGVSp', 'HGVSp_Short', 'Transcript_ID', 'Exon_Number', 't_depth', 't_ref_count', 't_alt_count', 'n_depth', 'n_ref_count', 'n_alt_count', 'all_effects', 'Allele', 'Gene', 'Feature', 'Feature_type', 'One_Consequence', 'Consequence', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 'DISTANCE', 'TRANSCRIPT_STRAND', 'SYMBOL', 'SYMBOL_SOURCE', 'HGNC_ID', 'BIOTYPE', 'CANONICAL', 'CCDS', 'ENSP', 'SWISSPROT', 'TREMBL', 'UNIPARC', 'UNIPROT_ISOFORM', 'RefSeq', 'MANE', 'APPRIS', 'FLAGS', 'SIFT', 'PolyPhen', 'EXON', 'INTRON', 'DOMAINS', '1000G_AF', '1000G_AFR_AF', '1000G_AMR_AF', '1000G_EAS_AF', '1000G_EUR_AF', '1000G_SAS_AF', 'ESP_AA_AF', 'ESP_EA_AF', 'gnomAD_AF', 'gnomAD_AFR_AF', 'gnomAD_AMR_AF', 'gnomAD_ASJ_AF', 'gnomAD_EAS_AF', 'gnomAD_FIN_AF', 'gnomAD_NFE_AF', 'gnomAD_OTH_AF', 'gnomAD_SAS_AF', 'MAX_AF', 'MAX_AF_POPS', 'gnomAD_non_cancer_AF', 'gnomAD_non_cancer_AFR_AF', 'gnomAD_non_cancer_AMI_AF', 'gnomAD_non_cancer_AMR_AF', 'gnomAD_non_cancer_ASJ_AF', 'gnomAD_non_cancer_EAS_AF', 'gnomAD_non_cancer_FIN_AF', 'gnomAD_non_cancer_MID_AF', 'gnomAD_non_cancer_NFE_AF', 'gnomAD_non_cancer_OTH_AF', 'gnomAD_non_cancer_SAS_AF', 'gnomAD_non_cancer_MAX_AF_adj', 'gnomAD_non_cancer_MAX_AF_POPS_adj', 'CLIN_SIG', 'SOMATIC', 'PUBMED', 'TRANSCRIPTION_FACTORS', 'MOTIF_NAME', 'MOTIF_POS', 'HIGH_INF_POS', 'MOTIF_SCORE_CHANGE', 'miRNA', 'IMPACT', 'PICK', 'VARIANT_CLASS', 'TSL', 'HGVS_OFFSET', 'PHENO', 'GENE_PHENO', 'CONTEXT', 'tumor_bam_uuid', 'normal_bam_uuid', 'case_id', 'GDC_FILTER', 'COSMIC', 'hotspot', 'RNA_Support', 'RNA_depth', 'RNA_ref_count', 'RNA_alt_count', 'callers']

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
