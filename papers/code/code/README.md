# Cloned Code Repositories

This directory contains cloned repositories with implementations relevant to artificial token language research.

## Repository 1: Byte Latent Transformer (BLT) - Official Implementation

- **URL:** https://github.com/facebookresearch/blt
- **Purpose:** Official implementation of the BLT architecture from Meta Research
- **Location:** `code/blt/`
- **License:** CC-BY-NC-4.0

### Description
The official code for the paper "Byte Latent Transformer: Patches Scale Better Than Tokens". This is the primary implementation for the state-of-the-art byte-level language model that matches tokenization-based LLM performance at scale.

### Key Features
- Training and inference code for BLT models
- Pre-trained model weights (1B and 7B parameters)
- Entropy-based dynamic patching
- Cross-attention mechanisms for byte-to-patch conversion
- Hash n-gram embeddings

### Key Files
- `blt/modeling/blt.py` - Main BLT model architecture
- `blt/modeling/entropy_model.py` - Entropy model for patching
- `blt/demo.py` - Simple generation demo
- `blt/download_blt_weights.py` - Download pre-trained weights

### Quick Start
```bash
cd code/blt
python download_blt_weights.py
python demo.py "An artificial token language"
```

### Usage for Research
This is the **most critical** repository for your research. Use it to:
1. Understand the BLT architecture implementation
2. Study entropy-based patching strategies
3. Analyze cross-attention mechanisms
4. Examine hash n-gram embedding implementation
5. Compare with your custom tokenization approaches

### Notes
- Requires CUDA-capable GPU for training
- Based on Meta Lingua framework
- HuggingFace model weights available at `facebook/blt`

---

## Repository 2: MEGABYTE - PyTorch Implementation

- **URL:** https://github.com/lucidrains/MEGABYTE-pytorch
- **Purpose:** Community implementation of the MEGABYTE architecture
- **Location:** `code/MEGABYTE-pytorch/`
- **License:** MIT

### Description
Implementation of "MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers". This is a predecessor to BLT that uses fixed-stride patching instead of dynamic patching.

### Key Features
- Multi-scale decoder architecture
- Fixed byte patching
- Local and global transformer models
- Handles sequences up to 1M bytes

### Key Files
- `megabyte_pytorch/megabyte_pytorch.py` - Main architecture
- `setup.py` - Installation configuration

### Quick Start
```bash
cd code/MEGABYTE-pytorch
pip install megabyte-pytorch

# Example usage
python -c "
from megabyte_pytorch import MEGABYTE

model = MEGABYTE(
    num_tokens = 256,          # 256 bytes
    dim = (512, 256),          # transformer dimensions
    depth = (6, 4),            # transformer depths
    max_seq_len = (512, 4),    # sequence lengths
    flash_attn = True          # use flash attention
)
"
```

### Usage for Research
Use as a **baseline** to compare against BLT:
1. Study fixed-stride patching vs. dynamic patching
2. Understand multiscale transformer architecture
3. Implement baseline comparisons in experiments
4. Analyze computational efficiency differences

### Notes
- Simpler architecture than BLT (good for understanding fundamentals)
- Mentioned in BLT paper as baseline
- Does not use cross-attention or hash n-grams

---

## Repository 3: minbpe - Educational BPE Implementation

- **URL:** https://github.com/karpathy/minbpe
- **Purpose:** Clean, minimal implementation of Byte Pair Encoding
- **Location:** `code/minbpe/`
- **License:** MIT

### Description
Minimal, clean code for the Byte Pair Encoding (BPE) algorithm by Andrej Karpathy. This is the algorithm used by GPT, Llama, and other modern LLMs for tokenization.

### Key Features
- Basic BPE implementation
- Regex-based BPE (GPT-2 style)
- GPT-4 compatible tokenizer
- Educational with thorough comments

### Key Files
- `minbpe/base.py` - Base tokenizer class
- `minbpe/basic.py` - Basic BPE algorithm
- `minbpe/regex.py` - Regex-based BPE (GPT-2 style)
- `minbpe/gpt4.py` - GPT-4 compatible tokenizer
- `exercise.md` - Educational exercises

### Quick Start
```bash
cd code/minbpe
pip install -e .

# Train a tokenizer
python -c "
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
text = 'Your training text here'
tokenizer.train(text, vocab_size=512)
tokenizer.save('tokenizer')

# Use tokenizer
ids = tokenizer.encode('Hello world')
text = tokenizer.decode(ids)
"
```

### Usage for Research
Use to **understand traditional tokenization**:
1. Implement BPE baselines for comparison
2. Study compression ratios of BPE
3. Analyze vocabulary construction
4. Compare byte-level encoding vs. BPE
5. Educational reference for tokenization theory

### Notes
- Very clean, educational code
- Good starting point for understanding tokenization
- Can be used to implement baselines
- Only ~400 lines of Python

---

## Repository 4: SentencePiece - Google's Tokenizer

- **URL:** https://github.com/google/sentencepiece
- **Purpose:** Production-ready tokenization library with BPE and Unigram support
- **Location:** `code/sentencepiece/`
- **License:** Apache 2.0

### Description
Google's unsupervised text tokenizer and detokenizer. Supports BPE, Unigram, character, and word models. Language-independent and widely used in production systems.

### Key Features
- Multiple tokenization algorithms (BPE, Unigram, Char, Word)
- Language-independent
- Subword regularization for robustness
- BPE-dropout for improved accuracy
- Fast C++ implementation with Python bindings

### Key Files
- `src/sentencepiece_trainer.h` - Training interface
- `src/sentencepiece_processor.h` - Tokenization interface
- `python/` - Python bindings
- `doc/` - Documentation

### Quick Start
```bash
cd code/sentencepiece
pip install sentencepiece

# Train a tokenizer
python -c "
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='input.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe'  # or 'unigram'
)

# Use tokenizer
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
ids = sp.encode('Hello world', out_type=int)
text = sp.decode(ids)
"
```

### Usage for Research
Use for **robust baselines**:
1. Train production-quality BPE tokenizers
2. Implement Unigram language model baselines
3. Compare regularization techniques
4. Evaluate multilingual tokenization
5. Study vocabulary size effects

### Notes
- Production-ready, battle-tested code
- Used by T5, BERT, and many other models
- Supports advanced features like subword regularization
- Can process multilingual data effectively

---

## Additional Recommended Repositories (Not Cloned)

### HuggingFace Tokenizers
- **URL:** https://github.com/huggingface/tokenizers
- **Purpose:** Fast, flexible tokenization library
- **Why not cloned:** Can be installed via pip
- **Install:** `pip install tokenizers`
- **Use for:** Production-grade tokenization, benchmarking

### GitHub rust-gems
- **URL:** https://github.com/github/rust-gems
- **Purpose:** Ultra-fast BPE tokenizer in Rust
- **Why not cloned:** Specialized use case
- **Use for:** Performance benchmarking

---

## Comparison Table

| Repository | Architecture | Patching | Speed | Use Case |
|------------|-------------|----------|-------|----------|
| **BLT** | Byte-level, dynamic | Entropy-based | Medium | Primary research implementation |
| **MEGABYTE** | Byte-level, fixed | Fixed-stride | Medium | Baseline comparison |
| **minbpe** | Subword | N/A | Fast | Educational, baselines |
| **SentencePiece** | Subword | N/A | Very Fast | Production baselines |

---

## Recommended Workflow

1. **Start with minbpe** to understand traditional tokenization
2. **Study MEGABYTE** to understand multiscale byte-level architectures
3. **Deep dive into BLT** for state-of-the-art byte-level modeling
4. **Use SentencePiece** for robust baseline comparisons

---

## Installation Requirements

### BLT
```bash
cd code/blt
pip install -r requirements.txt
# Requires: PyTorch, transformers, datasets, flash-attn
```

### MEGABYTE
```bash
pip install megabyte-pytorch
# Requires: PyTorch, einops
```

### minbpe
```bash
cd code/minbpe
pip install -e .
# Requires: regex
```

### SentencePiece
```bash
pip install sentencepiece
# Or build from source:
cd code/sentencepiece
mkdir build && cd build
cmake ..
make -j $(nproc)
sudo make install
```

---

## Key Takeaways for Experiment Runner

1. **BLT is the primary reference** - Most advanced byte-level approach
2. **MEGABYTE is important baseline** - Shows improvement of dynamic over fixed patching
3. **minbpe is educational** - Understand what you're improving upon
4. **SentencePiece for production baselines** - Fair comparison with established methods

---

## Research Questions to Explore

Using these repositories, investigate:

1. **Patching Strategies**
   - How does entropy-based patching (BLT) compare to fixed-stride (MEGABYTE)?
   - Can you design better patching heuristics?

2. **Vocabulary Design**
   - How do hash n-grams (BLT) compare to fixed vocabularies (BPE)?
   - What's the optimal vocabulary size for different approaches?

3. **Architectural Choices**
   - Do cross-attention mechanisms significantly improve performance?
   - What's the right balance between local and global models?

4. **Efficiency Metrics**
   - Bits-per-byte vs. perplexity vs. downstream task performance
   - Inference speed vs. model quality trade-offs

5. **Multilingual Performance**
   - How do different tokenization approaches handle non-Latin scripts?
   - Can byte-level models reduce multilingual inequity?

---

## Notes

- All repositories are cloned with `--depth 1` to save space
- Check individual repository README files for detailed documentation
- Some repositories may require specific CUDA versions or hardware
- License compatibility: Check licenses before using code in publications
