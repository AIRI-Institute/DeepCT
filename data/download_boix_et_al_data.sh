dataDir="./Boix_data"
mkdir -p $dataDir

wget --directory-prefix $dataDir https://personal.broadinstitute.org/cboix/epimap/metadata/main_metadata_table.tsv
wget --directory-prefix $dataDir --execute="robots = off" --mirror --no-parent -nd --wait=5 -A "FINAL*.bigWig" https://epigenome.wustl.edu/epimap/data/observed/

#process data
python process_boix_et_al_data.py Boix_data/ Boix_data/main_metadata_table.tsv --out Boix_data_processed/
cat Boix_data_processed/*.bed | sort -k 1,1 -k2,2n > Boix_data_processed/all_data_sorted.bed
