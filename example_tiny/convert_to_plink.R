devtools::install_github("uqrmaie1/admixtools")

# Copy files to .geno and .ind and .snp
file.copy("HumanOrigins249_tiny.snp", "copy.snp")
file.copy("HumanOrigins249_tiny.fam", "copy.ind")
file.copy("HumanOrigins249_tiny.eigenstratgeno", "copy.geno")

admixtools::eigenstrat_to_plink(
  inpref = "copy",
  outpref = "plink",
  verbose = TRUE
)

file.remove("copy.snp")
file.remove("copy.ind")
file.remove("copy.geno")
