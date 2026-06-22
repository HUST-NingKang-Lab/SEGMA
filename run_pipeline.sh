#!/usr/bin/env bash

python amp_sigma_pipeline.py \
  --input candidate_amps.csv \
  --training-csv all_peptide_structure_seqs_for_training.csv \
  --esmfold-model models/ESMFold \
  --esm-model models/esm2_650M \
  --saprot-model models/SaProt \
  --foldseek-bin tools/foldseek \
  --output outputs/predictions.csv

python predict_from_structures.py \
  --pdb-dir path/to/structures \
  --training-csv all_peptide_structure_seqs_for_training.csv \
  --model-dir outputs/model \
  --esm-model models/esm2_650M \
  --saprot-model models/SaProt \
  --foldseek-bin tools/foldseek \
  --output outputs/predictions_from_pdb.csv

python predict_from_structure_table.py \
  --structure-table path/to/foldseek_structure_seqs.csv \
  --training-csv all_peptide_structure_seqs_for_training.csv \
  --model-dir outputs/model \
  --esm-model models/esm2_650M \
  --saprot-model models/SaProt \
  --output outputs/predictions_from_structure_table.csv