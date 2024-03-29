# DrosophilaDenovoGene

## 1. ProcessingWholeGenomeAlignments
 - `bedAggregation.py`   Aggregate tiny gapped alignments generated by halLiftover in progressive cactus
 - `ortholog_mapping.py`   Get the aligned orthologs in progressive cactus
 - `ortholog_relations.py`   Get the ortholog information from the pairwise aligned orthologs identified from ortholog_mapping.py
 - `find_last_orf.py`  Find genes that are aligned to unannotated regions in their outgroup species
 - `check_homologs.py`   Exclude genes that are aligned to unannotated regions, but also have homologs in other chromosome regions in outgroup species.
 - `extract_raw_hit.py`  Extract raw hits from aggregated bed files and perform spliced alignments
 - `extract_alnscore_genewise.py`  Extract spliced alignment scores and get final de novo gene candidates

## 2. trRosetta
 - `fasta2seq.py`  Python script to handle protein/DNA sequence files
 - `build_MSA.py`  To search through different protein sequence database and generate multiple sequence alignments (MSA).
 - `reformat.py`  To extract sub-regions of MSA
 - `trRosetta.sh`  To run trRosetta predictions

## 3. MDsimulations
 - `densitypeaks.py`  To perform density-peaks clustering on MD trajectories
 - `densitypeaks.combineReps.py`  To perform density-peaks clustering on MD trajectories
