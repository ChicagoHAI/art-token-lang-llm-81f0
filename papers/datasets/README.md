# Datasets for Artificial Token Language Research

This directory contains datasets for the research project on training LLMs with compact, artificial token languages. Data files are NOT committed to git due to size. Follow the download instructions below.

## Overview

The research requires datasets for:
1. **Language Modeling** - Training and evaluating tokenization approaches
2. **Multilingual Evaluation** - Testing tokenization efficiency across languages
3. **Character-Level Tasks** - Evaluating byte/character-level understanding
4. **Benchmark Evaluation** - Standard NLP tasks for model comparison

---

## Dataset 1: WikiText-103

### Overview
- **Source**: https://huggingface.co/datasets/Salesforce/wikitext
- **Size**: ~103M tokens, ~500MB
- **Format**: HuggingFace Dataset
- **Task**: Language modeling
- **Splits**: train (1,801,350 tokens), validation (217,646 tokens), test (245,569 tokens)
- **License**: Creative Commons Attribution-ShareAlike

### Description
WikiText-103 is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. It's a standard benchmark for language modeling that tests long-term dependencies.

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext-103")
```

**Alternative - Manual Download:**
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip -d datasets/wikitext-103/
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-103")
```

### Sample Data
```
= Valkyria Chronicles III =
Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game...
```

### Notes
- Contains long-form articles suitable for testing long-range dependencies
- Pre-tokenized with spaces and punctuation
- Good for comparing different tokenization approaches

---

## Dataset 2: enwiki8 (Hutter Prize)

### Overview
- **Source**: http://mattmahoney.net/dc/enwik8.zip
- **Size**: 100MB (exactly 100,000,000 bytes)
- **Format**: Raw text file (byte-level)
- **Task**: Byte-level language modeling
- **License**: Wikipedia license

### Description
The first 100 million bytes of a Wikipedia XML dump. This is the standard benchmark for byte-level language models. Contains 205 unique byte values.

### Download Instructions

**Direct download:**
```bash
cd datasets
mkdir -p enwiki8
wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip -d enwiki8/
```

### Loading the Dataset

```python
with open("datasets/enwiki8/enwik8", "rb") as f:
    data = f.read()
# First 90M bytes for train, next 5M for valid, last 5M for test
train_data = data[:90000000]
valid_data = data[90000000:95000000]
test_data = data[95000000:]
```

### Sample Data (first 100 bytes)
```
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.3/" xmlns:xsi="http://www.w3.org/200
```

### Notes
- **Critical for byte-level research** - Standard benchmark mentioned in BLT and other byte-level papers
- Contains XML markup, providing diverse byte patterns
- Exactly 100M bytes makes it easy to compare compression ratios
- 205 unique tokens (bytes)

---

## Dataset 3: text8

### Overview
- **Source**: http://mattmahoney.net/dc/text8.zip
- **Size**: 100MB (100,000,000 bytes)
- **Format**: Raw text file (cleaned)
- **Task**: Character-level language modeling
- **License**: Wikipedia license

### Description
Derived from enwiki8 with all XML removed and text lowercased to only have 26 English letters plus spaces (27 characters total).

### Download Instructions

**Direct download:**
```bash
cd datasets
mkdir -p text8
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip -d text8/
```

### Loading the Dataset

```python
with open("datasets/text8/text8", "r") as f:
    data = f.read()
# Split: first 90M for train, next 5M for valid, last 5M for test
train_data = data[:90000000]
valid_data = data[90000000:95000000]
test_data = data[95000000:]
```

### Sample Data
```
anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy...
```

### Notes
- Simpler than enwiki8 - only 27 unique characters
- Good for testing character-level models without XML complexity
- Standardizes text for fair comparison

---

## Dataset 4: The Pile (Sample)

### Overview
- **Source**: https://huggingface.co/datasets/EleutherAI/pile
- **Size**: 825GB full (we'll use a sample)
- **Format**: HuggingFace Dataset
- **Task**: General language modeling
- **Splits**: train, validation, test
- **License**: Various (see dataset card)

### Description
The Pile is a 825 GiB diverse, open-source language modeling dataset consisting of 22 smaller, high-quality datasets combined. Mentioned in the BLT paper and other tokenization research.

### Download Instructions

**Download a sample (recommended for initial experiments):**
```python
from datasets import load_dataset

# Load only a subset (e.g., 10% of train split)
dataset = load_dataset("EleutherAI/pile", split="train[:10%]", streaming=True)

# Or download specific subsets
dataset = load_dataset("EleutherAI/pile", split="train", streaming=False)
# Save to disk
dataset.save_to_disk("datasets/pile_sample")
```

**For full dataset:**
```bash
# Warning: 825GB!
# Best to use streaming mode or download specific components
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/pile_sample")
```

### Notes
- Contains diverse domains: academic, code, books, web text, etc.
- Mentioned in "Tokenization Is More Than Compression" paper
- Use streaming mode for large-scale experiments
- Average 3.41 bytes per token across different subsets

---

## Dataset 5: FLORES-200

### Overview
- **Source**: https://huggingface.co/datasets/facebook/flores
- **Size**: ~100MB
- **Format**: HuggingFace Dataset
- **Task**: Multilingual machine translation evaluation
- **Splits**: dev, devtest
- **Languages**: 200+ languages
- **License**: CC-BY-SA 4.0

### Description
FLORES-200 is a multilingual parallel corpus covering 200+ languages. Crucial for evaluating tokenization efficiency across languages, especially low-resource ones. Mentioned in BLT paper Table 4.

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

dataset = load_dataset("facebook/flores", "all")
dataset.save_to_disk("datasets/flores-200")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/flores-200")

# Access specific language pairs
for example in dataset['dev']:
    print(example['eng_Latn'])  # English
    print(example['ara_Arab'])  # Arabic
    print(example['hin_Deva'])  # Hindi
```

### Sample Data
```json
{
  "id": 1,
  "eng_Latn": "The cat sat on the mat.",
  "ara_Arab": "جلست القطة على السجادة.",
  "hin_Deva": "बिल्ली चटाई पर बैठ गई।",
  "zho_Hans": "猫坐在垫子上。"
}
```

### Notes
- **Critical for multilingual evaluation** - Tests tokenization efficiency across scripts
- Covers low-resource languages (Armenian, Amharic, Assamese, Bengali, Georgian, etc.)
- Parallel sentences enable direct comparison of token counts across languages
- Used in BLT paper to evaluate low-resource translation

---

## Dataset 6: C4 (Colossal Clean Crawled Corpus)

### Overview
- **Source**: https://huggingface.co/datasets/allenai/c4
- **Size**: ~750GB (en), multilingual versions available
- **Format**: HuggingFace Dataset
- **Task**: General language modeling
- **License**: ODC-BY

### Description
C4 is a colossal, cleaned version of Common Crawl's web crawl corpus. It's used as a standard pretraining dataset. The "Tokenization Is More Than Compression" paper used C4 for training.

### Download Instructions

**Using HuggingFace (streaming recommended):**
```python
from datasets import load_dataset

# Stream to avoid downloading all 750GB
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Or download a specific subset
dataset = load_dataset("allenai/c4", "en", split="train[:1%]")
dataset.save_to_disk("datasets/c4_sample")
```

**For multilingual C4 (mC4):**
```python
dataset = load_dataset("allenai/c4", "multilingual", split="train", streaming=True)
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/c4_sample")
```

### Notes
- Used in many tokenization papers for pretraining
- Contains web text from Common Crawl
- Multilingual version (mC4) available for 101 languages
- Use streaming mode to avoid downloading entire dataset

---

## Dataset 7: CUTE Benchmark

### Overview
- **Source**: https://huggingface.co/datasets/lutzkuen/CUTE
- **Size**: ~1MB
- **Format**: HuggingFace Dataset
- **Task**: Character-level understanding evaluation
- **License**: Research use

### Description
CUTE (Character-level Understanding Test Evaluation) benchmark from the paper "CUTE: Measuring LLMs' Understanding of Their Tokens". Tests character manipulation, orthographic similarity, and semantic tasks. Mentioned in BLT paper Table 3.

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

dataset = load_dataset("lutzkuen/CUTE")
dataset.save_to_disk("datasets/cute")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cute")
```

### Sample Tasks
```json
{
  "task": "substitute_word",
  "prompt": "Question: Substitute 'and' with 'internet' in 'She went to the kitchen and saw two cereals.'\nAnswer:",
  "answer": "She went to the kitchen internet saw two cereals."
},
{
  "task": "swap_char",
  "prompt": "Question: Swap 'h' and 'a' in 'that'.\nAnswer:",
  "answer": "taht"
}
```

### Notes
- **Critical for evaluating byte-level models** - Tests character-level understanding
- BLT achieved 54.1% vs Llama 3's 27.5%
- Tasks include: character substitution, word manipulation, spelling, orthography
- Small size makes it easy to include

---

## Dataset 8: HellaSwag (Noisy Variants)

### Overview
- **Source**: https://huggingface.co/datasets/Rowan/hellaswag
- **Size**: ~50MB
- **Format**: HuggingFace Dataset
- **Task**: Commonsense reasoning + robustness evaluation
- **License**: MIT

### Description
HellaSwag is a commonsense NLI benchmark. For robustness testing, create noised versions using strategies from BLT paper: AntSpeak, Drop, RandomCase, Repeat, UpperCase.

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

dataset = load_dataset("Rowan/hellaswag")
dataset.save_to_disk("datasets/hellaswag")
```

### Creating Noisy Variants

```python
import random

def add_antspeak_noise(text):
    """Convert to uppercase, space-separated characters"""
    return ' '.join(text.upper())

def add_drop_noise(text, drop_rate=0.1):
    """Randomly remove 10% of characters"""
    chars = list(text)
    return ''.join(c for c in chars if random.random() > drop_rate)

def add_randomcase_noise(text):
    """Random uppercase/lowercase"""
    return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)

def add_repeat_noise(text, repeat_rate=0.2):
    """Repeat 20% of characters up to 4 times"""
    result = []
    for c in text:
        result.append(c)
        if random.random() < repeat_rate:
            result.extend([c] * random.randint(1, 3))
    return ''.join(result)

def add_uppercase_noise(text):
    """Convert all to uppercase"""
    return text.upper()
```

### Notes
- Used to test robustness to input noise
- BLT showed significant improvements over Llama 3 on noisy variants
- Create noisy versions on-the-fly during evaluation

---

## Summary Table

| Dataset | Size | Type | Primary Use | Priority |
|---------|------|------|-------------|----------|
| WikiText-103 | 500MB | Text | Language modeling benchmark | High |
| enwiki8 | 100MB | Bytes | Byte-level LM standard benchmark | **Critical** |
| text8 | 100MB | Characters | Character-level LM | High |
| The Pile (sample) | Variable | Text | Diverse language modeling | Medium |
| FLORES-200 | 100MB | Parallel text | Multilingual evaluation | **Critical** |
| C4 (sample) | Variable | Web text | Pretraining corpus | Medium |
| CUTE | 1MB | Prompts | Character understanding | **Critical** |
| HellaSwag | 50MB | Commonsense QA | Robustness testing | High |

## Quick Start Script

To download the most critical datasets quickly:

```bash
# Create download script
cat > datasets/download.sh << 'EOF'
#!/bin/bash
pip install datasets

python3 << 'PYTHON'
from datasets import load_dataset

# 1. WikiText-103
print("Downloading WikiText-103...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext-103")

# 2. FLORES-200
print("Downloading FLORES-200...")
dataset = load_dataset("facebook/flores", "all")
dataset.save_to_disk("datasets/flores-200")

# 3. CUTE
print("Downloading CUTE...")
dataset = load_dataset("lutzkuen/CUTE")
dataset.save_to_disk("datasets/cute")

# 4. HellaSwag
print("Downloading HellaSwag...")
dataset = load_dataset("Rowan/hellaswag")
dataset.save_to_disk("datasets/hellaswag")

print("Core datasets downloaded successfully!")
PYTHON

# 5. enwiki8 (byte-level)
echo "Downloading enwiki8..."
mkdir -p enwiki8
cd enwiki8
wget -q http://mattmahoney.net/dc/enwik8.zip
unzip -q enwik8.zip
cd ..

# 6. text8
echo "Downloading text8..."
mkdir -p text8
cd text8
wget -q http://mattmahoney.net/dc/text8.zip
unzip -q text8.zip
cd ..

echo "All critical datasets downloaded!"
EOF

chmod +x datasets/download.sh
```

Then run:
```bash
./datasets/download.sh
```

## Notes for Experiment Runner

1. **Start with small datasets** (enwiki8, text8, CUTE) for initial experiments
2. **Use streaming** for large datasets (C4, The Pile) to avoid storage issues
3. **FLORES-200 is critical** for multilingual tokenization efficiency evaluation
4. **Create noisy variants** of evaluation sets on-the-fly for robustness testing
5. **Byte-level metrics** - Use bits-per-byte (BPB) instead of perplexity for fair comparison

## Citation

If using these datasets, please cite the original sources:
- WikiText: Merity et al. (2016)
- enwiki8/text8: Hutter Prize
- The Pile: Gao et al. (2020)
- FLORES: Goyal et al. (2022)
- C4: Raffel et al. (2020)
- CUTE: Edman et al. (2024)
- HellaSwag: Zellers et al. (2019)
