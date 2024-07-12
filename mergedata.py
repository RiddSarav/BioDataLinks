# import pandas as pd

# # Define file paths
# file1_path = "file_with_single_column.tsv"
# file2_path = "file_with_multiple_columns.tsv"
# output_path = "combined_data.tsv"

# # Read the files using pandas.read_csv with sep='\t' for tab separation
# df1 = pd.read_csv(file1_path, sep='\t', header=None, names=["col1"])
# df2 = pd.read_csv(file2_path, sep='\t')

# # Combine the DataFrames by adding the single column from df1 as a new column in df2
# df_combined = pd.concat([df1, df2], axis=1)

# # Save the combined data to a new TSV file
# df_combined.to_csv(output_path, sep='\t', index=False)

# print("Files combined successfully!")
import pandas as pd

# File paths
file_path1 = "/Users/riddhishsaravanan/BioData/Clinical/balls.tsv"
file_path2 = "/Users/riddhishsaravanan/BioData/Clinical/clinical.tsv"  # Replace with the actual path of the second file
output_file_path = "/Users/riddhishsaravanan/BioData/Clinical/merged_file.tsv"

# Read both TSV files
df1 = pd.read_csv(file_path1, sep='\t')
df2 = pd.read_csv(file_path2, sep='\t')

# Concatenate the DataFrames horizontally
merged_df = pd.concat([df1, df2], axis=1)

# Save the merged DataFrame to a new TSV file
merged_df.to_csv(output_file_path, sep='\t', index=False)

print(f"The files have been merged and saved to '{output_file_path}'.")
