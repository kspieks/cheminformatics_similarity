
CHEM_SIM="../../cheminformatics_similarity"

out_file='ESOL_similarity.pkl'

data_path1="data/ESOL/delaney.csv"
data_path2="data/ESOL/delaney.csv"

fp_type="count"
metric="tanimoto"

n_cpus=2
n_cpus_featurize=2

python $CHEM_SIM/calculate_chemical_similarity.py \
--out_file $out_file \
--data_path1 $data_path1 \
--data_path2 $data_path2 \
--fp_type $fp_type \
--metric $metric \
--n_cpus $n_cpus \
--n_cpus_featurize $n_cpus_featurize
