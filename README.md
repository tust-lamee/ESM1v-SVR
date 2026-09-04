# ESM-1v-SVR: Topt Prediction Model

This repository contains the code and data for the Topt prediction model developed in our study. The model integrates the ESM-1v protein language model with support vector regression (SVR) to predict enzyme optimal catalytic temperature (Topt) from protein sequences.

---

## Repository Structure

---

## Dataset Description

### Excel File (`dataset.xlsx`)

The Excel file contains three sheets:

| Sheet | Description |
|-------|-------------|
| **Sheet1: Topt metadata from UniProt** | Contains raw data retrieved from UniProt, including protein sequences, UniProt accession numbers, source organisms, and experimentally determined Topt values. |
| **Sheet2: Topt-CDHIT 0.8** | Contains sequences with experimental Topt annotations after redundancy removal using CD-HIT at 80% sequence identity threshold. |
| **Sheet3: OGT-CDHIT 0.8** | Contains sequences with OGT labels from the Engqvist 2018 public dataset after redundancy removal using CD-HIT at 80% sequence identity threshold. |

---

### `OGT_data` Folder

This folder contains FASTA files of protein sequences from organisms with known optimal growth temperatures (OGT). These sequences were randomly extracted from the genomes of organisms grouped by temperature category:

| Folder | Temperature Range |
|--------|-------------------|
| `0-20/` | ≤ 20°C |
| `20-30/` | 20–30°C |
| `40-60/` | 40–60°C |
| `60-80/` | 60–80°C |
| `80-100/` | 80–100°C |

The OGT labels used in training were derived from these source organisms.

---

### Final Training Dataset (`9581`)

The file named `9581` contains the final curated dataset of **9,581 non-redundant protein sequences** used for model training.  
The file format is: `protein_sequence label`

---

## Model Description

Our model employs **ESM-1v** (Evolutionary Scale Modeling version 1) to encode each protein sequence into a **1,280-dimensional** feature vector via mean pooling across residue-level representations. The extracted features are then standardized using Z-score normalization and fed into a **support vector regression (SVR)** model with an **RBF kernel** to predict Topt.

**Key Features:**

- **Input:** Protein sequence only
- **Output:** Predicted Topt (°C)
- **Feature dimension:** 1,280
- **Regression model:** SVR with RBF kernel
- **Training data:** 9,581 sequences covering a broad temperature range (-20°C to 120°C)

---

## Usage

### Step 1: Extract Features for Training Sequences

Extract ESM-1v embeddings from the 9,581 training sequences:

```bash
python ESM-1v.py --input 9581 --output train_features.csv
```
### Step 2: Train the SVR Model

Train the SVR model using the extracted features and temperature labels:
```bash

python SVR.py --features train_features.csv --output svr_model.pkl
```
This generates the trained model file (svr_model.pkl) used for prediction.
### Step 3: Extract Features for Unknown Sequences

For new sequences with unknown Topt values:
```bash

python ESM-1v.py --input unknown_sequences.fasta --output unknown_features.csv
```
### Step 4: Predict Topt

Predict Topt for unknown sequences using the trained model:
```bash

python predicted_temperatures.py \
  --model svr_model.pkl \
  --features unknown_features.csv \
  --output predictions.csv
```
The output file (predictions.csv) contains the predicted Topt values for each sequence.
Data Source

    Topt data: Retrieved from UniProt database using the advanced search function with non-empty "Temperature dependence" fields. Proteins with experimentally determined optimal temperatures were selected and classified by their Topt values.

    OGT data: Obtained from Engqvist MKM. Correlating enzyme annotations with a large set of microbial growth temperatures reveals metabolic adaptations to growth at diverse temperatures. BMC Microbiol. 2018;18:177. The dataset includes OGT values for over 21,000 microorganisms. Protein sequences were randomly extracted from the genomes of organisms in each temperature category.

    CD‑HIT parameters: Sequence identity threshold of 80%, calculated as the percentage of identical residues over the aligned region relative to the shorter sequence, with default alignment coverage parameters.

Citation

If you use this code or dataset in your research, please cite our paper:

    Cai Y, Bian Y, Chen Y, Wang F, Zhao L, Peng C, Li Y, Lu F. Ancestral sequence reconstruction synergized with deep learning significantly enhances the thermophilicity and thermostability of the highly active alkaline protease AprE.

License

This project is licensed under the MIT License.
Contact

For questions or issues, please contact:

    Chong Peng: cpeng@tust.edu.cn
