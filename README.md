# Introduction

This repository provides a fine-tuned BERT-CRF model for Named Entity Recognition (NER) on Japanese pharmaceutical care records.  
The model is designed to extract symptoms and drug names from the free-text “Subjective” section of SOAP notes written by community pharmacists.

---

## Key Features

- Extracts symptoms and drug names from Japanese pharmaceutical care records
- Capable of extracting colloquial expressions from patient “Subjective” section
- Trained on the Pharmacist Complaint Annotation (PCA) dataset
- Outputs entity tags in XML-style format (e.g., `<d>`, `<m-key>`)

---

## Model Overview

| Item            | Detail                                                  |
|-----------------|---------------------------------------------------------|
| Base model      | `cl-tohoku/bert-base-japanese-v3`                       |
| Architecture    | BERT encoder + Conditional Random Field (CRF) layer     |
| Fine-tuning data| Pharmacist Complaint Annotation (PCA) dataset           |
| Max input length| 512 tokens                                              |
| Training device | NVIDIA RTX™ 4500 Ada                                    |
| Epochs          | 5                                                       |
| Batch size      | 32                                                      |
| Learning rate   | 1e-4                                                    |
| Decoding        | Viterbi algorithm for sequence prediction in CRF layer  |

---

## How to use

Download the following all files and put into the same folder.

(Note) The `test/` directory contains backup and maintenance versions of prediction scripts.  
These scripts are not used in the actual prediction pipeline.

```
newNERmodel/
├── predict_bertcrf.py                 # Prediction script
├── utils.py                           # Utility functions
├── label2id.pkl                       # Label-to-ID mapping file
├── data/
│   └── text.txt                       # Input test text file
└── checkpoint-150/                    # Folder containing the trained model
    ├── pytorch_model.bin              # Model weights
    ├── config.json                    # Model configuration
    ├── tokenizer_config.json          # Tokenizer configuration
    ├── special_tokens_map.json        # Special token mappings
    └── vocab.txt                      # Vocabulary file
```

This repository uses **Git Large File Storage (LFS)** to manage the model file `pytorch_model.bin`.  
If you download the repository as a ZIP file, the model file will not be included.  
Please follow the steps below to properly clone the repository and download the LFS-managed files.

### 1. Install Git LFS

#### macOS

```bash
brew install git-lfs
git lfs install
```

#### Windows

```
Download and run the Git LFS installer from the official site:
https://git-lfs.github.com/
```
### 2. Clone the repository and pull LFS files

```bash
git clone https://github.com/keioDI/newNERmodel.git
cd newNERmodel
git lfs pull
```
Note: git lfs pull is required to download the actual model file (`pytorch_model.bin`).

You can use this model by running predict_bertcrf.py.

```bash
python3 predict_bertcrf.py
```

### Input Example

```
たまに足がふらっとする。薬が変わったからかな。あざが増えた気がするけど、ぶつけた覚えはない。
動悸が少し減った気はするけど、なんか全体的にだるい。時々、鼻血が出やすくなったように感じる。
```

### Output Example

```
たまに<d certainly="positive">足がふらっとする</d>。<m-key state="executed">薬</m-key>が変わったからかな。<d certainly="positive">あざ</d>が増えた気がするけど、ぶつけた覚えはない。
<d certainly="positive">動悸</d>が少し減った気はするけど、なんか全体的に<d certainly="positive">だるい</d>。時々、<d certainly="positive">鼻血</d>が出やすくなったように感じる。
```
