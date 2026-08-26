# Amharic ↔ Oromo Machine Translation

A neural machine translation system for translating **Amharic text into Oromo** using a custom **Transformer encoder-decoder architecture**.

The project focuses on building a data-driven translation model for Amharic–Oromo parallel text and evaluating its performance using the **BLEU (Bilingual Evaluation Understudy)** metric.

---

## 📌 Project Overview

This project develops a Transformer-based machine translation model for the Amharic and Oromo languages.

The system takes an Amharic sentence as input and generates its corresponding Oromo translation.

### Translation Direction

**Amharic → Oromo**

Example:

```text
Amharic:
አሁን በእርግጥ መጠጥ እፈልጋለሁ.

Oromo Reference:
amma dhugaatii baay'een barbaada.

Model Prediction:
dhuguma amma dhugaatii barbaada.
```

The project includes:

* Parallel corpus preparation
* Text preprocessing
* SentencePiece BPE tokenization
* Transformer model development
* Model training
* Validation using BLEU
* Final translation testing
* TensorBoard monitoring
* Weights & Biases experiment tracking

---

## 🧠 Model Architecture

The translation system uses a custom Transformer encoder-decoder architecture.

### Configuration

| Parameter              |                       Value |
| ---------------------- | --------------------------: |
| Model                  | Transformer Encoder-Decoder |
| `d_model`              |                         128 |
| Attention Heads        |                           4 |
| Encoder Layers         |                           2 |
| Decoder Layers         |                           2 |
| Feed-Forward Dimension |                         512 |
| Vocabulary Size        |                       8,000 |
| Training Epochs        |                          20 |
| Total Parameters       |                   4,005,696 |
| Model Size             |                   ~15.28 MB |

The trained model contains approximately **4 million trainable parameters**.

The final output layer uses an 8,000-token vocabulary.

---

## 📚 Dataset

The project uses a parallel corpus containing aligned Amharic and Oromo sentences.

Each training example consists of:

```text
Amharic sentence → Oromo sentence
```

The dataset is processed before training to improve consistency between the source and target languages.

### Data Preparation

The preprocessing pipeline includes operations such as:

* Removing missing values
* Removing duplicate sentence pairs
* Cleaning unnecessary whitespace
* Normalizing text
* Preparing source and target sentences
* Tokenizing the processed text

---

## 🔤 Tokenization

The project uses **SentencePiece BPE (Byte Pair Encoding)** tokenization.

The vocabulary size used for the Transformer model is:

```text
8,000 tokens
```

BPE allows the model to represent both common words and previously unseen or rare words using smaller subword units.

---

## 🏗️ Transformer Components

The model contains the major components of the Transformer architecture:

### Encoder

The encoder processes the Amharic input sequence and creates contextual representations.

It uses:

* Multi-head self-attention
* Feed-forward neural networks
* Residual connections
* Layer normalization
* Dropout

### Decoder

The decoder generates the Oromo translation one token at a time.

It uses:

* Masked self-attention
* Encoder-decoder attention
* Feed-forward networks
* Residual connections
* Layer normalization
* Dropout

---

## 🚀 Training

The model was trained for:

```text
20 epochs
```

During training, the following metrics were monitored:

* Training loss
* Validation loss
* Masked accuracy
* Validation BLEU

The training logs show that the training loss decreased from approximately **5.53 at the beginning of epoch 1** to approximately **1.49 by epoch 20**.
Masked accuracy also improved throughout training, reaching approximately **70.77%** in epoch 20.

---

## 📊 Evaluation

The main evaluation metric is **BLEU**.

BLEU compares generated translations against reference translations and measures the degree of overlap between them.

### Validation BLEU

The validation BLEU score varied considerably across epochs.

Some recorded validation BLEU scores were:

| Epoch | Validation BLEU |
| ----: | --------------: |
|     1 |           29.07 |
|     2 |           53.73 |
|     3 |           32.17 |
|     4 |            8.64 |
|     5 |           39.76 |
|     6 |           36.60 |
|     7 |            0.00 |
|     8 |           20.41 |
|     9 |           35.36 |
|    10 |           35.93 |
|    11 |           50.00 |
|    12 |           50.00 |
|    13 |          100.00 |
|    14 |           21.02 |
|    15 |           30.74 |
|    16 |           30.21 |
|    17 |           25.41 |
|    18 |           14.06 |
|    19 |           26.08 |
|    20 |            0.00 |

These fluctuations indicate that validation BLEU was unstable during training.

### Final Test Result

```text
Final Test BLEU: 31.04
```

---

## 🧪 Translation Examples

### Example 1

**Amharic**

```text
አሁን በእርግጥ መጠጥ እፈልጋለሁ.
```

**Reference**

```text
amma dhugaatii baay'een barbaada.
```

**Prediction**

```text
dhuguma amma dhugaatii barbaada.
```

---

### Example 2

**Amharic**

```text
ቶም ተናዘዘ።
```

**Reference**

```text
toom himateera.
```

**Prediction**

```text
toom ni aare.
```

---

### Example 3

**Amharic**

```text
የቤት ስራዬን እንዲረዳኝ ቶምን ጠየቅሁት።
```

**Reference**

```text
hojii manaa koo akka na gargaaru toom gaafadhe.
```

**Prediction**

```text
hojii manaa koo akka gargaaru toom gaafadhe.
```

---

### Example 4

**Amharic**

```text
ሚሊየነር አይደለሁም።
```

**Reference**

```text
ani miliyeenara miti.
```

**Prediction**

```text
ani miliyeenara miti.
```

---

## 💾 Model

The trained Transformer model was saved as:

```text
amh_omo_transformer.keras
```

The training output reports the model location as:

```text
/home/genesistwo/newe/addis/amh_omo_transformer.keras
```

---

## 📈 Experiment Tracking

Training experiments were monitored using **Weights & Biases (W&B)**.

The following metrics were tracked:

* Epoch
* Learning rate
* Training loss
* Training masked accuracy
* Validation loss
* Validation masked accuracy
* Validation BLEU

TensorBoard was also used for monitoring training.

The recorded TensorBoard log directory was:

```text
/home/genesistwo/newe/addis/logs/fit/20260629-132045
```

---

## 🛠️ Technologies

The project uses:

* Python
* TensorFlow / Keras
* SentencePiece
* NumPy
* Pandas
* TensorBoard
* Weights & Biases
* Transformer architecture
* BLEU evaluation

---

## 📁 Suggested Project Structure

```text
amharic-oromo-translation/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── preprocessing/
│   └── preprocess.py
│
├── tokenizer/
│   ├── tokenizer.model
│   └── tokenizer.vocab
│
├── model/
│   └── transformer.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── inference/
│   └── translate.py
│
├── notebooks/
│
└── outputs/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd amharic-oromo-translation
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

After preparing the dataset and tokenizer, run:

```bash
python train.py
```

The training process will train the Transformer model and evaluate it on the validation set.

---

## 🔎 Translation

After training, the model can be used to translate an Amharic sentence:

```text
Amharic input
      ↓
Preprocessing
      ↓
SentencePiece tokenizer
      ↓
Transformer Encoder
      ↓
Transformer Decoder
      ↓
Oromo output
```

---

## 📌 Current Results

| Metric                |      Result |
| --------------------- | ----------: |
| Model                 | Transformer |
| Vocabulary            |       8,000 |
| Parameters            |   4,005,696 |
| Training Epochs       |          20 |
| Final Training Loss   |      1.4891 |
| Final Validation Loss |      1.7304 |
| Final Masked Accuracy |      70.77% |
| Final Test BLEU       |   **31.04** |

The final epoch recorded a training loss of approximately 1.4891, validation loss of approximately 1.7304, and masked accuracy of approximately 70.77%.

---

## ⚠️ Limitations

The current model still has several areas that can be improved.

The validation BLEU score fluctuated significantly between epochs, including some epochs with a BLEU score of 0.00.
Some translations are accurate, while others contain incorrect words or meanings.

For example, the model correctly translated one test sentence:

```text
ani miliyeenara miti.
```

but produced an incorrect translation for another sentence:

```text
toom ni aare.
```

instead of the reference:

```text
toom himateera.
```

---

## 🔮 Future Improvements

Future work can focus on:

* Increasing the size and quality of the parallel corpus
* Improving data cleaning and normalization
* Investigating the instability of validation BLEU
* Improving tokenization
* Hyperparameter tuning
* Increasing model capacity
* Using learning-rate scheduling
* Experimenting with different Transformer configurations
* Improving decoding strategies
* Evaluating with additional metrics such as chrF and TER
* Testing on a larger independent test set
* Comparing the custom Transformer with pretrained multilingual translation models

---

## 👩‍💻 Author

**Blen Gebre**

Software Engineering — AI Stream

Addis Ababa University

---

## 📄 Project Status

**Research / Development**

The current system demonstrates the feasibility of neural machine translation between Amharic and Oromo and achieved a final test BLEU score of **31.04**.

The project is continuing to be improved through better data, model optimization, and evaluation.
