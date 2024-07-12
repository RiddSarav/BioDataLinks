# Install and load required packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("minfi")
BiocManager::install("illuminaio")

library(minfi)
library(illuminaio)

# Set the working directory to the folder containing your IDAT files
setwd("/Users/riddhishsaravanan/BioData/ment")

# Verify the current working directory
current_dir <- getwd()
print(current_dir)

# List files in the directory to ensure IDAT files are present
files <- list.files()
print(files)

# Read the IDAT files with custom pattern
RGset <- read.metharray.exp(base = current_dir, pattern = "_Grn.idat$")

# Check the data
head(RGset)

# Preprocess the data (optional step)
Mset <- preprocessRaw(RGset)

# Output the first few rows of the MethylationSet object
head(getBeta(Mset))
