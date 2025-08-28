# PriBic-CCA

### Description
**Pri**vate **Bic**lustering Analysis with **C**heng and **C**hurch **A**lgorithm: malicious secure gene expression data analysis by biclustering algorithm namely Cheng and Church Algorithm using secure Multiparty Computation (MPC) schemes, including Replicate Secret Sharing (RSS) and Function Secret Sharing (FSS) in Python under the MIT license.

### Dependencies
We apply:  
1. [Funshade](https://github.com/ibarrond/funshade) for IC (Interval Containment) gate of function secret sharing scheme in 2PC protocol to perform comparison
2. [Sycret](https://github.com/OpenMined/sycret) for DPF (Distributed Point Function) gate of function secret sharing scheme in 2PC protocol to perform equality check
3. [Biclustlib](https://github.com/padilha/biclustlib) for implementation of the original CCA, yeast cell cycle dataset and accuracy measures

## Datasets
Our input data are *yeast cell cycle* taken from [Tavazoie et al. (1999)](https://pubmed.ncbi.nlm.nih.gov/10391217/) and human expression data taken from [Alizadeh et al. (2000)](https://www.nature.com/articles/35000501), which were used in the original study by [Cheng and Church](https://www.researchgate.net/profile/George_Church/publication/2329589_Biclustering_of_Expression_Data/links/550c04030cf2063799394f5e.pdf);

## External Evaluation Measure
To measure the similarity of encrypted biclusters with non-encrypted version, we use [Liu Wang](https://academic.oup.com/bioinformatics/article/23/1/50/189870) match score, along with [Prelic](https://academic.oup.com/bioinformatics/article/22/9/1122/200492) relevance as external evaluation measures.

## Important Project Contents
- `cca.py` contains implementation of secured CCA utilising MPC schemes 
- `origCCA.py` contains implementation of original CCA
- `accuracy.py` contains implementation of accuracy measures
- `test_cca.py` contains sample implementation of both secured and original algorithms, datasets and evaluation measures

### Code Author and Contributor
Shokofeh VahidianSadegh, and Alberto Ibarrondo

_The code accompanying under review manuscript._
