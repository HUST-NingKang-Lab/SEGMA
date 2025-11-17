# SEGMA
We introduced **structure-aware extremophile genome mining for antimicrobial peptides (SEGMA)**, a deep-learning based framework used to systematically mine and optimize AMPs (exrtremocins) from extreme environments on a global scale.

We applied [APEX 1.1](https://gitlab.com/machine-biology-group-public/apex-pathogen) to predict MIC of extremoxins , [Foldseek](https://github.com/steineggerlab/foldseek) to geneate peptides' 3Di sequences .

**Preprint**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.13.688364v1) The paper is contiually updated! Welcome to follow our new results.

### Python Dependencies
Install required packages with:

```bash
chmod +x download.sh
```

```bash
./download.sh
```

Before running **SEGMA**, convert amino acid sequences into **aa-3Di** sequences using the following command:

```bash
python get_seqs.py -i your_sequences.fasta -o candidate.csv
```

To optimize extremocins with **SEGMA**, run the following command:

```bash
python optimize.py -i candidate.csv -o candidate_optimize.csv
```

> **Note:** This project is continuously maintained and updated!

## Maintainers

| Name           | Email                                     | Affiliation                                                |
|----------------|-------------------------------------------|-------------------------------------------------------------|
| **Zixin Kang**    | [29590kang@gmail.com](mailto:29590kang@gmail.com)     | Graduate Student, School of Life Science and Technology, HUST |
| **Haohong Zhang** | [haohongzh@gmail.com](mailto:haohongzh@gmail.com)     | PhD Student, School of Life Science and Technology, HUST     |
| **Kang Ning**     | [ningkang@hust.edu.cn](mailto:ningkang@hust.edu.cn)   | Professor, School of Life Science and Technology, HUST       |
