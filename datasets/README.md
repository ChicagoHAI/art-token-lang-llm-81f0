# Downloaded Datasets

This directory contains datasets for the research project on artificial token languages for efficient LLMs. **Data files are NOT committed to git due to size.** Follow the download instructions below.

---

## Dataset 1: TinyStories

### Overview
- **Source**: [roneneldan/TinyStories on HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Size**: ~2.1M training stories (~500 MB), 21,990 validation stories
- **Format**: HuggingFace Dataset (text)
- **Task**: Language modeling, text generation
- **Splits**: train (~2.1M stories), validation (~22K stories)
- **License**: Apache 2.0
- **Why this dataset**:
  - Used in vocabulary compression paper (2411.06371)
  - Perfect for testing compact tokenization on controlled vocabulary
  - Small vocabulary enables faster experimentation
  - Good for training small LLMs (125M-1B params)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Download full dataset
dataset = load_dataset("roneneldan/TinyStories")
dataset.save_to_disk("datasets/tinystories")

# Or download just V2 (GPT-4 only, higher quality)
dataset_v2 = load_dataset("roneneldan/TinyStories", "TinyStoriesV2-GPT4")
dataset_v2.save_to_disk("datasets/tinystories_v2")
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/tinystories")

# Access train split
train_data = dataset['train']
print(train_data[0]['text'])
```

### Sample Data

Example story from the dataset:
```text
Once upon a time, there was a little car named Beep. Beep loved to go fast. One day, Beep was going very fast on a big track. Beep went zoom, zoom, zoom! Beep's friend, a little girl named Mia, came to watch. Mia liked to clap when Beep went fast...
```

### Notes
- Dataset contains short stories using vocabulary that 3-4 year olds understand
- Two versions: V1 (GPT-3.5 + GPT-4) and V2 (GPT-4 only - higher quality)
- Average story length: ~200-300 tokens
- Excellent for testing tokenization efficiency on simple but natural language

---

## Dataset 2: WikiText-103

### Overview
- **Source**: [Salesforce/wikitext on HuggingFace](https://huggingface.co/datasets/Salesforce/wikitext)
- **Size**: ~103M training tokens, ~240K validation tokens
- **Format**: HuggingFace Dataset (text)
- **Task**: Language modeling
- **Splits**: train (28,475 articles), validation (60 articles), test (60 articles)
- **License**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- **Why this dataset**:
  - Standard benchmark for language modeling
  - Used extensively in tokenization research
  - 110x larger than Penn Treebank
  - Clean, high-quality Wikipedia text

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Download WikiText-103 (word-level)
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext/wikitext-103-v1")

# Or download raw version (character-level)
dataset_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
dataset_raw.save_to_disk("datasets/wikitext/wikitext-103-raw-v1")
```

**Alternative (direct download):**
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip -d datasets/wikitext/
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext/wikitext-103-v1")

# Access train split
train_data = dataset['train']
print(train_data[0]['text'])
```

### Sample Data

Example from WikiText-103:
```text
= Valkyria Chronicles III =

Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable...
```

### Notes
- Contains verified Good and Featured Wikipedia articles
- Two versions: processed (word-level) and raw (character-level)
- Raw version recommended for tokenization experiments
- Pre-tokenized into articles for convenient processing

---

## Dataset 3: enwik8

### Overview
- **Source**: [Matt Mahoney's Large Text Compression Benchmark](http://mattmahoney.net/dc/textdata.html)
- **Source (HuggingFace)**: [LTCB/enwik8](https://huggingface.co/datasets/LTCB/enwik8)
- **Size**: 100 MB (100,000,000 bytes) uncompressed
- **Format**: Raw text (UTF-8 XML)
- **Task**: Compression benchmarking, character-level language modeling
- **Splits**: Typically first 90M for train, 5M for validation, 5M for test
- **License**: Public domain (Wikipedia content)
- **Why this dataset**:
  - Standard compression benchmark
  - Used in multiple tokenization papers (BLT, ByT5, MEGABYTE)
  - Tests byte-level and character-level modeling
  - Compact size for quick experiments

### Download Instructions

**Direct download (recommended for simplicity):**
```bash
cd datasets/enwik8
wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip
```

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("LTCB/enwik8")
dataset.save_to_disk("datasets/enwik8_hf")
```

**Using Python script:**
```python
import urllib.request
import zipfile

# Download
url = "http://mattmahoney.net/dc/enwik8.zip"
urllib.request.urlretrieve(url, "datasets/enwik8/enwik8.zip")

# Unzip
with zipfile.ZipFile("datasets/enwik8/enwik8.zip", 'r') as zip_ref:
    zip_ref.extractall("datasets/enwik8/")
```

### Loading the Dataset

Once downloaded:
```python
# Load raw file
with open('datasets/enwik8/enwik8', 'r', encoding='utf-8') as f:
    data = f.read()

# Standard split
train_data = data[:90000000]  # First 90M bytes
valid_data = data[90000000:95000000]  # Next 5M bytes
test_data = data[95000000:]  # Last 5M bytes
```

### Sample Data

Example from enwik8 (Wikipedia XML):
```xml
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.3/" version="0.3">
  <siteinfo>
    <sitename>Wikipedia</sitename>
    <base>http://en.wikipedia.org/wiki/Main_Page</base>
  </siteinfo>
  <page>
    <title>Anarchism</title>
    <text xml:space="preserve">
      '''Anarchism''' is a [[political philosophy]]...
```

### Notes
- **Already downloaded** in `datasets/enwik8/` directory
- Contains first 100MB of English Wikipedia XML dump (March 2006)
- UTF-8 encoded XML with primarily English text
- Standard splits: 90M train / 5M valid / 5M test
- Excellent for testing byte-level tokenization and compression

---

## Dataset 4: C4 (Colossal Clean Crawled Corpus)

### Overview
- **Source**: [allenai/c4 on HuggingFace](https://huggingface.co/datasets/allenai/c4)
- **Size**: ~750 GB (English), 108 languages available in mC4
- **Format**: HuggingFace Dataset (text)
- **Task**: Large-scale language model pretraining
- **Splits**: train (364M documents), validation (364K documents)
- **License**: ODC-BY
- **Why this dataset**:
  - Used to train T5 and many other LLMs
  - Largest scale for testing tokenization scalability
  - Clean, high-quality web text
  - Multiple language variants available

### Download Instructions

**IMPORTANT**: C4 is very large (~750GB). Consider using streaming or downloading subsets.

**Streaming (recommended for large dataset):**
```python
from datasets import load_dataset

# Stream without downloading entire dataset
dataset = load_dataset("allenai/c4", "en", streaming=True)

# Process in batches
for batch in dataset['train'].take(1000):
    print(batch['text'])
```

**Download subset:**
```python
from datasets import load_dataset

# Download small subset for testing
dataset = load_dataset("allenai/c4", "en", split="train[:1%]")
dataset.save_to_disk("datasets/c4/c4_1percent")

# Or validation set (smaller)
dataset = load_dataset("allenai/c4", "en", split="validation")
dataset.save_to_disk("datasets/c4/c4_validation")
```

**Download full dataset (requires ~800GB+ disk space):**
```python
from datasets import load_dataset

# WARNING: This will download ~750GB
dataset = load_dataset("allenai/c4", "en")
dataset.save_to_disk("datasets/c4/c4_full")
```

### Loading the Dataset

For streaming:
```python
from datasets import load_dataset
dataset = load_dataset("allenai/c4", "en", streaming=True)
for example in dataset['train']:
    print(example['text'])
    break
```

For downloaded subset:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/c4/c4_1percent")
print(dataset[0]['text'])
```

### Sample Data

Example from C4:
```text
Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.
```

### Notes
- **Do NOT download full dataset unless necessary** - use streaming or subsets
- Cleaned version of Common Crawl (no spam, duplicates, offensive content)
- Five variants: en, en.noclean, en.noblocklist, realnewslike, multilingual (mC4)
- Recommended: Start with validation split (smaller) or 1% sample
- For production experiments, use streaming to avoid disk space issues

---

## Dataset 5: GLUE & SuperGLUE (Optional - for downstream evaluation)

### Overview
- **Source**: [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) and [super_glue](https://huggingface.co/datasets/super_glue)
- **Size**: Varies by task (typically 1-100MB per task)
- **Format**: HuggingFace Dataset (text pairs, classification)
- **Task**: Downstream NLP task evaluation
- **License**: Various (mostly permissive)
- **Why this dataset**:
  - Standard benchmarks for evaluating language understanding
  - Test if compact tokenization preserves reasoning ability
  - Complement perplexity metrics with task performance

### Download Instructions

**GLUE:**
```python
from datasets import load_dataset

# Download all GLUE tasks
glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']

for task in glue_tasks:
    dataset = load_dataset("nyu-mll/glue", task)
    dataset.save_to_disk(f"datasets/glue/{task}")
```

**SuperGLUE:**
```python
from datasets import load_dataset

# Download all SuperGLUE tasks
superglue_tasks = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

for task in superglue_tasks:
    dataset = load_dataset("super_glue", task)
    dataset.save_to_disk(f"datasets/superglue/{task}")
```

### Notes
- Download only if planning downstream task evaluation
- Each task is relatively small (1-100MB)
- GLUE and SuperGLUE provide complementary tasks
- Useful for final validation of tokenization quality

---

## Summary Table

| Dataset | Size | Primary Use | Status |
|---------|------|-------------|--------|
| TinyStories | ~500 MB | Small-scale LM training, controlled vocab | Ready to download |
| WikiText-103 | ~500 MB | Standard LM benchmark | Ready to download |
| enwik8 | 100 MB | Compression benchmark, byte-level LM | **Downloaded** ✓ |
| C4 | ~750 GB | Large-scale pretraining | Streaming recommended |
| GLUE/SuperGLUE | ~1-5 GB | Downstream evaluation | Optional |

---

## Recommended Download Priority

### For Initial Experiments (Small Scale):
1. **enwik8** (already downloaded) - Start here for compression tests
2. **TinyStories** - Best for quick LLM training experiments
3. **WikiText-103 validation** - Standard benchmark evaluation

### For Full-Scale Experiments:
1. **TinyStories** - Full training
2. **WikiText-103** - Full benchmark
3. **C4 validation** or **1% sample** - Large-scale test
4. **GLUE** (selected tasks) - Downstream evaluation

### For Production Research:
- **C4 full** with streaming - Large-scale pretraining
- **WikiText-103** - Standard reporting
- **SuperGLUE** - Comprehensive evaluation

---

## Storage Requirements

- **Minimal setup** (enwik8 + TinyStories): ~600 MB
- **Standard setup** (+ WikiText-103): ~1.1 GB
- **Full setup** (+ C4 sample + GLUE): ~5-10 GB
- **Production setup** (+ C4 full): ~800 GB

---

## Quick Start Script

```python
from datasets import load_dataset

# 1. Download TinyStories (recommended starting point)
print("Downloading TinyStories...")
tinystories = load_dataset("roneneldan/TinyStories")
tinystories.save_to_disk("datasets/tinystories")

# 2. Download WikiText-103
print("Downloading WikiText-103...")
wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
wikitext.save_to_disk("datasets/wikitext/wikitext-103-raw-v1")

# 3. enwik8 already downloaded!
print("enwik8 already available in datasets/enwik8/")

# 4. Download C4 validation (small subset)
print("Downloading C4 validation set...")
c4_val = load_dataset("allenai/c4", "en", split="validation")
c4_val.save_to_disk("datasets/c4/c4_validation")

print("Dataset downloads complete!")
```

---

## Notes for Experiment Runner

1. **enwik8 is ready to use** - no download needed
2. **Start with TinyStories** for fast iteration
3. **Use streaming for C4** to avoid storage issues
4. **All datasets support HuggingFace's datasets library** for easy loading
5. **Consider tokenizer vocabulary size** when choosing datasets:
   - TinyStories: ~10K unique words (small vocab)
   - WikiText-103: ~250K unique tokens
   - C4: ~1M+ unique tokens
6. **Validation sets are smaller** - use them first for quick tests

---

## References

- TinyStories: https://arxiv.org/abs/2305.07759
- WikiText: Merity et al., "Pointer Sentinel Mixture Models" (2016)
- enwik8: Large Text Compression Benchmark
- C4: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2019)
- GLUE: Wang et al., "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding" (2018)
- SuperGLUE: Wang et al., "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems" (2019)
