# PriBic-CCA

### Description
**Pri**vate **Bic**lustering Analysis with **C**heng and **C**hurch **A**lgorithm: malicious secure gene expression data analysis by biclustering algorithm namely Cheng and Church Algorithm using secure Multiparty Computation (MPC) schemes, including Replicate Secret Sharing (RSS) and Function Secret Sharing (FSS) in Python under the MIT license.

### Dependencies
We apply:  
1. [Funshade](https://github.com/ibarrond/funshade) for IC (Interval Containment) gate of function secret sharing scheme in 2PC protocol to perform comparison
2. [Sycret](https://github.com/OpenMined/sycret) for DPF (Distributed Point Function) gate of function secret sharing scheme in 2PC protocol to perform equality check
3. [Biclustlib](https://github.com/padilha/biclustlib) for implementation of the original CCA, yeast cell cycle dataset and accuracy measures

## External Evaluation Measure
To measure the similarity of encrypted biclusters with non-encrypted version, we use Liu Wang match score, along with Prelic relevance as external evaluation measures.

## Important Project Contents
- `cca.py` contains implementation of secured CCA utilising MPC schemes 
- `origCCA.py` contains implementation of original CCA
- `accuracy.py` contains implementation of accuracy measures
- `test_cca.py` contains sample implementation of both secured and original algorithms, gene expression data sets and evaluation measures

### Code Author and Contributor
Shokofeh VahidianSadegh, and Alberto Ibarrondo

_The code accompanying in prepration manuscript._
