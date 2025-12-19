#!/bin/bash

ls ./cameo_target > name.txt
cut -d'.' -f1 name.txt > list.txt
rm name.txt

for name in $(cat list.txt)
do
	./scripts/fix_numbering.pl ./cameo_predict/${name}_relaxed.pdb ./cameo_target/${name}.pdb
	sed -i '/.\{26\}X.*/d' cameo_predict/${name}_relaxed.pdb.fixed
	mv cameo_predict/${name}_relaxed.pdb.fixed cameo_predict/${name}_relaxed.pdb_processed
done
cp ./cameo_predict/*processed ../../openfold_0416/alphafold_0416/
rm list.txt
