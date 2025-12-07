# Cloned Code Repositories

This directory contains relevant code repositories for the artificial token language research project.

---

## Repository 1: Byte Latent Transformer (BLT)

- **URL**: https://github.com/facebookresearch/blt
- **Organization**: Meta FAIR (Facebook AI Research)
- **Purpose**: State-of-the-art byte-level language model implementation
- **Location**: `code/blt/`
- **Key Features**:
  - First byte-level LLM to match tokenization-based performance at scale
  - Dynamic patch-based architecture with entropy-adaptive segmentation
  - Scaling up to 8B parameters
  - Pre-trained models and training code available

### Key Files
- `blt/modeling.py` - BLT model architecture
- `blt/data.py` - Data loading and patching logic
- `blt/train.py` - Training scripts
- `README.md` - Setup and usage instructions

### Notes
- **Highly relevant** for understanding tokenizer-free approaches
- Demonstrates that byte-level models can match subword performance
- Can be used as baseline implementation
- Requires PyTorch and substantial compute for training

### Usage Example
```bash
cd code/blt
# Follow installation instructions in README.md
pip install -e .

# Example: Load pre-trained BLT model
python -c "from blt import BLT; model = BLT.from_pretrained('blt-8b')"
```

---

## Repository 2: MEGABYTE

- **URL**: https://github.com/shjwudp/megabyte
- **Author**: shjwudp (Community implementation)
- **Purpose**: Multi-scale transformer for tokenization-free modeling
- **Location**: `code/megabyte/`
- **Key Features**:
  - PyTorch implementation of MEGABYTE architecture
  - Hierarchical patch-based approach (local + global models)
  - Sub-quadratic attention complexity
  - Supports million-byte sequences

### Key Files
- `megabyte/model.py` - MEGABYTE architecture
- `megabyte/data.py` - Data processing and patching
- `examples/` - Training and inference examples
- `README.md` - Documentation and setup

### Notes
- Foundation for BLT architecture
- Good for understanding hierarchical tokenization-free approaches
- Community implementation (not official Meta code)
- Useful for experimenting with multi-scale architectures

### Usage Example
```bash
cd code/megabyte
pip install -r requirements.txt

# Train on custom data
python train.py --data_path /path/to/data
```

---

## Repository 3: ByT5

- **URL**: https://github.com/google-research/byt5
- **Organization**: Google Research
- **Purpose**: Token-free T5 variant operating on UTF-8 bytes
- **Location**: `code/byt5/`
- **Key Features**:
  - Official Google implementation
  - Pre-trained byte-level models available
  - Competitive with subword models on many tasks
  - Multilingual and noise-robust

### Key Files
- `README.md` - Setup and model download instructions
- `byt5/` - Model code (uses T5X framework)
- Pre-trained checkpoints available via HuggingFace

### Notes
- **Production-ready** byte-level baseline
- Pre-trained models available (small, base, large, xl, xxl)
- Uses Google's T5X/JAX framework (not PyTorch)
- Excellent baseline for comparing against artificial token languages

### Usage Example
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained ByT5
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# Process bytes directly
text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
```

---

## Repository 4: SentencePiece

- **URL**: https://github.com/google/sentencepiece
- **Organization**: Google
- **Purpose**: Unsupervised text tokenizer supporting BPE and Unigram LM
- **Location**: `code/sentencepiece/`
- **Key Features**:
  - Language-independent tokenization
  - Supports BPE, Unigram LM, and character/word tokenization
  - Used in many production LLMs (T5, ALBERT, XLNet, etc.)
  - Fast C++ implementation with Python bindings

### Key Files
- `src/` - C++ implementation
- `python/` - Python bindings
- `doc/` - Documentation
- `README.md` - Installation and usage

### Notes
- **Industry standard** for subword tokenization
- Essential for training baseline tokenizers
- Can train custom vocabularies on our artificial language
- Comparison baseline for our research

### Usage Example
```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='data.txt',
    model_prefix='tokenizer',
    vocab_size=8000,
    model_type='bpe'  # or 'unigram'
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode('Hello, world!', out_type=str)
```

---

## Repository 5: HuggingFace Tokenizers

- **URL**: https://github.com/huggingface/tokenizers
- **Organization**: HuggingFace
- **Purpose**: Fast tokenizers library in Rust with Python bindings
- **Location**: `code/tokenizers/`
- **Key Features**:
  - Extremely fast tokenization (Rust implementation)
  - Supports BPE, WordPiece, Unigram, and custom algorithms
  - Easy training of custom tokenizers
  - Compatible with HuggingFace Transformers

### Key Files
- `bindings/python/` - Python API
- `tokenizers/src/` - Rust implementation
- `examples/` - Training and usage examples
- `README.md` - Documentation

### Notes
- **Recommended** for implementing custom tokenization
- Much faster than pure Python implementations
- Easy to experiment with custom vocabularies
- Can be extended for artificial token languages

### Usage Example
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Create and train custom tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=10000, special_tokens=["[PAD]", "[UNK]"])

# Train on files
tokenizer.train(files=["data.txt"], trainer=trainer)

# Save
tokenizer.save("custom_tokenizer.json")
```

---

## Summary Table

| Repository | Purpose | Language | Pre-trained Models | Relevance |
|------------|---------|----------|-------------------|-----------|
| BLT | SOTA byte-level LLM | PyTorch | Yes | ★★★★★ |
| MEGABYTE | Multi-scale byte modeling | PyTorch | No (community) | ★★★★☆ |
| ByT5 | Production byte-level T5 | JAX/T5X | Yes | ★★★★★ |
| SentencePiece | Subword tokenization | C++/Python | No (training tool) | ★★★★☆ |
| HuggingFace Tokenizers | Fast tokenization library | Rust/Python | No (training tool) | ★★★★★ |

---

## Recommended Usage Priority

### For Understanding Byte-Level Approaches:
1. **ByT5** - Start here, production-ready with pre-trained models
2. **BLT** - Latest SOTA, understand dynamic patching
3. **MEGABYTE** - Historical context, hierarchical approach

### For Tokenization Experiments:
1. **HuggingFace Tokenizers** - Fast, flexible, easy to customize
2. **SentencePiece** - Industry standard, good baseline
3. **BLT patching** - Inspiration for dynamic token boundaries

### For Baselines:
1. **ByT5 small** - Byte-level baseline
2. **SentencePiece BPE** - Standard subword baseline
3. **SentencePiece Unigram** - Alternative subword baseline
4. **BLT** - SOTA byte-level baseline

---

## Installation Quick Start

```bash
# Install all Python dependencies
pip install torch transformers datasets sentencepiece tokenizers

# Install SentencePiece (if C++ build needed)
cd code/sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
cd ../../

# Install BLT
cd code/blt
pip install -e .
cd ../

# Install HuggingFace tokenizers (usually via pip is enough)
pip install tokenizers
```

---

## Next Steps for Experiment Runner

1. **Start with ByT5** as byte-level baseline using pre-trained models
2. **Use HuggingFace Tokenizers** to implement custom artificial token language
3. **Train baselines** with SentencePiece (BPE and Unigram)
4. **Reference BLT** for dynamic/learned tokenization approaches
5. **Compare** all approaches on same datasets (TinyStories, WikiText-103, enwik8)

---

## Key Insights from Code Review

### Design Patterns to Consider:
1. **Entropy-based segmentation** (BLT) - allocate compute where uncertainty is high
2. **Hierarchical processing** (MEGABYTE) - multi-scale representation
3. **Byte-level robustness** (ByT5) - handle any text without preprocessing
4. **Fast training** (HuggingFace Tokenizers) - critical for iteration speed
5. **Standard interfaces** (all) - compatible with HuggingFace ecosystem

### Implementation Considerations:
- All implementations use standard deep learning frameworks (PyTorch/JAX)
- Pre-trained models available for ByT5 and BLT (save training time)
- HuggingFace Tokenizers allows custom tokenization algorithms
- Can mix byte-level and token-level approaches

### Performance Benchmarks from Code:
- ByT5: Competitive with mT5 on multilingual tasks
- BLT: Matches LLaMA-3 scaling behavior
- MEGABYTE: State-of-the-art on ImageNet density estimation
- HuggingFace Tokenizers: 10-100x faster than Python implementations

---

## References

- BLT Paper: https://arxiv.org/abs/2412.09871
- MEGABYTE Paper: https://arxiv.org/abs/2305.07185
- ByT5 Paper: https://arxiv.org/abs/2105.13626
- SentencePiece Paper: https://arxiv.org/abs/1808.06226
- HuggingFace Tokenizers Docs: https://huggingface.co/docs/tokenizers
