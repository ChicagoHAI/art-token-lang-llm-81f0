# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "An Artificial Token Language for More Efficient LLMs". The project explores training large language models on compact, artificial token languages to reduce model size and computational requirements while maintaining or improving reasoning quality.

**Date:** December 7, 2025
**Research Domain:** NLP / Large Language Models
**Resource Gathering Phase:** Completed

---

## Papers

**Total papers downloaded:** 15

### Core Papers (Priority: Critical)

| Title | Authors | Year | File | arXiv | Key Contribution |
|-------|---------|------|------|-------|------------------|
| Byte Latent Transformer: Patches Scale Better Than Tokens | Pagnoni et al. (Meta) | 2024 | 2412.09871_byte_latent_transformer.pdf | 2412.09871 | First byte-level LLM to match token-based at scale; entropy-based dynamic patching |
| Tokenization Is More Than Compression | Schmidt et al. | 2024 | 2402.18376_tokenization_more_than_compression.pdf | 2402.18376 | Proves compression ≠ performance; PathPiece tokenizer |
| Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling | Huang et al. | 2025 | 2501.16975_over_tokenized_transformer.pdf | 2501.16975 | Decouples input/output vocabs; log-linear scaling |

### Important Papers (Priority: High)

| Title | Authors | Year | File | arXiv | Key Contribution |
|-------|---------|------|------|-------|------------------|
| SpaceByte: Towards Deleting Tokenization | Slagle | 2024 | 2404.14408_spacebyte.pdf | 2404.14408 | Space-based patching; hierarchical byte processing |
| MrT5: Dynamic Token Merging | Kallini et al. | 2024 | 2410.20771_mrt5.pdf | 2410.20771 | Learned gating for token deletion; 75% compression |
| T-FREE: Tokenizer-Free via Sparse Representations | Deiseroth et al. | 2024 | 2406.19223_tfree.pdf | 2406.19223 | Sparse character triplet embeddings; 85% param reduction |
| From Tokens to Characters | Vieira et al. | 2024 | 2412.03719_tokens_to_characters.pdf | 2412.03719 | Conversion algorithms for token→character |
| Training LLMs over Neurally Compressed Text | Lester et al. | 2024 | 2404.03626_neurally_compressed_text.pdf | 2404.03626 | Neural compression for tokenization |
| Hierarchical Autoregressive Transformers | Neitemeier et al. | 2025 | 2501.10322_hierarchical_transformers.pdf | 2501.10322 | Hybrid byte + word processing |

### Additional Papers (Priority: Medium)

| Title | File | arXiv | Notes |
|-------|------|-------|-------|
| BPE is Suboptimal for LM Pretraining | 2004.03720_bpe_suboptimal.pdf | 2004.03720 | Early BPE critique |
| Charformer | 2106.12672_charformer.pdf | 2106.12672 | Character transformer with gradient-based subword |
| Rethinking Tokenization | 2403.00417_rethinking_tokenization.pdf | 2403.00417 | General tokenization critique |
| Unpacking Tokenization | 2403.06265_unpacking_tokenization.pdf | 2403.06265 | Comprehensive analysis |
| A Theory of Tokenization | 2404.08335_theory_of_tokenization.pdf | 2404.08335 | Theoretical foundations |
| Retok: Efficient Tokenizer | 2410.04335_retok_efficient_tokenizer.pdf | 2410.04335 | Efficient tokenization approach |

See `papers/README.md` for detailed descriptions of each paper.

---

## Datasets

**Total datasets identified:** 8 (3 critical, 3 high priority, 2 medium priority)

| Name | Source | Size | Task | Priority | Status |
|------|--------|------|------|----------|--------|
| **enwiki8** | Hutter Prize | 100MB | Byte-level LM | **CRITICAL** | Ready to download |
| **FLORES-200** | HuggingFace | ~100MB | Multilingual eval | **CRITICAL** | Ready to download |
| **CUTE** | HuggingFace | ~1MB | Character understanding | **CRITICAL** | Ready to download |
| WikiText-103 | HuggingFace | ~500MB | Language modeling | High | Ready to download |
| text8 | Hutter Prize | 100MB | Char-level LM | High | Ready to download |
| HellaSwag | HuggingFace | ~50MB | Robustness | High | Ready to download |
| C4 (sample) | HuggingFace | Variable | Web text | Medium | Streaming recommended |
| The Pile (sample) | HuggingFace | Variable | Diverse LM | Medium | Streaming recommended |

### Critical Datasets Detail

**enwiki8 (Byte-level Standard)**
- 100,000,000 bytes exactly
- 205 unique bytes
- Standard benchmark mentioned in BLT, MrT5, SpaceByte papers
- Direct download: http://mattmahoney.net/dc/enwik8.zip

**FLORES-200 (Multilingual Evaluation)**
- 200+ languages with parallel sentences
- Used in BLT paper Table 4 for translation
- Tests tokenization efficiency across scripts
- Critical for evaluating multilingual equity

**CUTE (Character Understanding)**
- Character-level manipulation tasks
- BLT achieved 54.1% vs Llama 3's 27.5%
- Tests: substitution, orthography, spelling
- Small size (<1MB), easy to include

### Dataset Notes

- **Data files excluded from git** via datasets/.gitignore
- **Download instructions** provided in datasets/README.md
- **Quick start script** available: datasets/download.sh
- **Streaming mode** recommended for large datasets (C4, The Pile)

See `datasets/README.md` for complete download instructions and usage examples.

---

## Code Repositories

**Total repositories cloned:** 4

| Name | URL | Purpose | Location | License | Priority |
|------|-----|---------|----------|---------|----------|
| **BLT** | github.com/facebookresearch/blt | Official BLT implementation | code/blt/ | CC-BY-NC-4.0 | **CRITICAL** |
| **MEGABYTE** | github.com/lucidrains/MEGABYTE-pytorch | Baseline byte-level model | code/MEGABYTE-pytorch/ | MIT | High |
| **minbpe** | github.com/karpathy/minbpe | Educational BPE implementation | code/minbpe/ | MIT | High |
| **SentencePiece** | github.com/google/sentencepiece | Production tokenizer | code/sentencepiece/ | Apache 2.0 | High |

### Repository Details

**BLT (Byte Latent Transformer)** - **PRIMARY REFERENCE**
- State-of-the-art byte-level architecture
- Entropy-based dynamic patching
- Hash n-gram embeddings
- Cross-attention mechanisms
- Pre-trained weights: 1B and 7B models
- *Use for: Understanding SOTA, implementing experiments*

**MEGABYTE** - **BASELINE**
- Fixed-stride patching (predecessor to BLT)
- Multiscale architecture
- Simpler than BLT (good for learning)
- *Use for: Baseline comparisons, understanding fundamentals*

**minbpe** - **EDUCATIONAL**
- Minimal, clean BPE implementation
- ~400 lines of Python
- Educational exercises included
- *Use for: Understanding traditional tokenization, implementing baselines*

**SentencePiece** - **PRODUCTION BASELINE**
- Google's battle-tested tokenizer
- Supports BPE, Unigram, Char, Word models
- Fast C++ with Python bindings
- *Use for: Robust baseline comparisons*

### Not Cloned (But Recommended)

- **HuggingFace Tokenizers**: `pip install tokenizers` - Production-grade, very fast
- **GitHub rust-gems**: Ultra-fast Rust BPE (4× faster than tiktoken)

See `code/README.md` for detailed setup instructions and usage examples.

---

## Resource Gathering Notes

### Search Strategy

**Literature Search:**
- arXiv: cs.CL, cs.LG categories (2023-2025)
- Papers with Code: "tokenization", "byte-level", "language models"
- Web search: "artificial token language", "byte-level transformers", "efficient tokenization"
- Citation tracking: References in BLT paper

**Dataset Search:**
- HuggingFace Hub: language modeling, multilingual, evaluation benchmarks
- Standard benchmarks: Hutter Prize (enwiki8, text8), FLORES, CUTE
- Papers: Datasets mentioned in key papers (especially BLT)

**Code Search:**
- GitHub: Official paper repositories
- Implementation quality: Star count, documentation, maintenance
- License compatibility: Prefer permissive (MIT, Apache) where possible

### Selection Criteria

**Papers:**
- Recent (2024-2025 preferred for SOTA)
- High relevance to artificial token languages
- Byte-level or alternative tokenization focus
- arXiv papers with available PDFs
- Highly cited or from reputable institutions

**Datasets:**
- Established benchmarks for comparability
- Diverse (language modeling, multilingual, character-level)
- Reasonable size (prefer <1GB for core datasets)
- Used in key papers (especially BLT)
- Clear licensing

**Code:**
- Official implementations preferred
- Active maintenance
- Clear documentation
- Permissive licensing
- Educational value (clean code)

### Challenges Encountered

1. **Large dataset sizes:** C4 (750GB), The Pile (825GB)
   - **Solution:** Use streaming mode, download samples only

2. **Some papers lack code:** SpaceByte, MrT5, T-FREE
   - **Solution:** Implement from paper descriptions, use BLT as reference

3. **Pre-existing papers in directory:** Some papers already downloaded
   - **Solution:** Preserved existing downloads, added new ones

4. **Git repository size:** Datasets and large repos could bloat repo
   - **Solution:** .gitignore for data, --depth 1 for repos

### Gaps and Workarounds

**Gaps:**
1. **No official SpaceByte code** - Will need to implement from paper
2. **No official MrT5 code** - Can adapt from T5 + dynamic deletion
3. **Limited artificial token language prior work** - This is novel research!

**Workarounds:**
1. BLT provides comprehensive baseline for byte-level approach
2. Combination of papers provides theoretical foundation
3. Codebases provide implementation patterns to adapt

---

## Recommendations for Experiment Design

Based on gathered resources, here are concrete recommendations:

### 1. Primary Dataset(s)

**Must use:**
- **enwiki8** (100MB) - Standard byte-level benchmark, enables comparison with all prior work
- **FLORES-200** - Critical for multilingual evaluation
- **CUTE** - Tests character-level understanding (key differentiator for byte-level models)

**Should use:**
- **WikiText-103** - Standard language modeling benchmark
- **HellaSwag (+ noisy variants)** - Robustness testing

**Optional (if resources permit):**
- **C4 sample** - Web text diversity
- **The Pile subset** - Domain diversity

### 2. Baseline Methods

**Essential baselines:**
1. **BLT (entropy patching)** - Current SOTA byte-level
2. **Llama 3 BPE tokenizer** - Strong token-based baseline
3. **MEGABYTE (fixed patching)** - Demonstrates value of dynamic patching

**Recommended baselines:**
4. **SentencePiece BPE** - Standard production approach
5. **Byte-level (no patching)** - Naive baseline

### 3. Evaluation Metrics

**Primary metrics (must report):**
- **Bits-per-byte (BPB)** - Tokenizer-independent performance measure
- **Training FLOPs** - Efficiency during training
- **Inference FLOPs** - Efficiency at deployment
- **Downstream accuracy** - At least 3 tasks (recommend: ARC, HellaSwag, MMLU)

**Secondary metrics (should report):**
- **Compression ratio** (bytes/token) - But remember: compression ≠ performance!
- **Robustness** - Accuracy on noisy inputs
- **Multilingual equity** - Token count variance across languages
- **Character-level accuracy** - CUTE benchmark score

### 4. Code to Adapt/Reuse

**Primary codebase:**
- **BLT repository** - Use as starting point for implementation
  - Entropy model training
  - Dynamic patching logic
  - Hash n-gram embeddings
  - Cross-attention mechanisms

**Baseline implementations:**
- **MEGABYTE** - Fixed patching baseline
- **minbpe** - Traditional BPE baseline
- **SentencePiece** - Production-quality baseline

**Utilities:**
- **HuggingFace datasets** - Data loading
- **HuggingFace tokenizers** - Baseline tokenizers

### 5. Experimental Design

**Phase 1: Reproduce Baselines**
1. Verify BLT results on enwiki8
2. Train BPE tokenizers with minbpe
3. Establish baseline performance

**Phase 2: Design Artificial Token Language**
1. Analyze BLT's hash n-gram approach
2. Design custom symbol system
3. Implement encoding/decoding

**Phase 3: Train and Evaluate**
1. Train models with artificial tokens
2. Compare against baselines on all metrics
3. Analyze what works and why

**Phase 4: Multilingual Testing**
1. Evaluate on FLORES-200
2. Measure tokenization efficiency across scripts
3. Test robustness with CUTE

---

## Quick Start Guide for Experiment Runner

### Step 1: Download Critical Datasets
```bash
cd datasets
./download.sh  # Downloads enwiki8, text8, FLORES-200, CUTE, WikiText, HellaSwag
```

### Step 2: Setup Code Repositories
```bash
# BLT
cd code/blt
pip install -r requirements.txt
python download_blt_weights.py

# minbpe (for baselines)
cd ../minbpe
pip install -e .

# SentencePiece
pip install sentencepiece
```

### Step 3: Run Baseline Experiments
```bash
# Test BLT
cd code/blt
python demo.py "Testing BLT"

# Train BPE tokenizer
cd code/minbpe
python train.py --input ../../datasets/enwiki8/enwik8 --vocab-size 8192
```

### Step 4: Implement Your Approach
- Start from BLT codebase
- Modify patching strategy for artificial tokens
- Compare against baselines

### Step 5: Evaluate
- BPB on enwiki8, text8
- Downstream tasks: ARC, HellaSwag, MMLU
- Multilingual: FLORES-200
- Character understanding: CUTE

---

## File Structure Summary

```
art-token-lang-llm-81f0/
├── papers/                          # 15 research papers (PDFs)
│   ├── README.md                    # Detailed paper descriptions
│   └── *.pdf                        # Downloaded papers
├── datasets/                        # Datasets (data excluded from git)
│   ├── .gitignore                   # Excludes data files
│   ├── README.md                    # Download instructions & usage
│   └── download.sh                  # Quick download script
├── code/                            # Cloned repositories
│   ├── README.md                    # Repository documentation
│   ├── blt/                         # BLT (Official, CRITICAL)
│   ├── MEGABYTE-pytorch/            # MEGABYTE baseline
│   ├── minbpe/                      # Educational BPE
│   └── sentencepiece/               # Google tokenizer
├── literature_review.md             # Comprehensive synthesis
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker (created when done)
```

---

## Timeline and Effort

**Resource gathering completed:** December 7, 2025
**Total time spent:** ~3 hours

**Breakdown:**
- Literature search and download: ~45 min
- Paper review and extraction: ~45 min
- Dataset identification and documentation: ~45 min
- Code repository search and cloning: ~30 min
- Documentation (README files, literature review, this catalog): ~45 min

**Papers:** 15 downloaded and documented
**Datasets:** 8 identified with complete instructions
**Repositories:** 4 cloned and documented
**Lines of documentation:** ~2,500+

---

## Next Steps for Experiment Runner

1. **Read literature_review.md** - Understand the research landscape
2. **Download critical datasets** - Run datasets/download.sh
3. **Setup BLT codebase** - Primary reference implementation
4. **Reproduce baseline** - Verify BLT results on enwiki8
5. **Design artificial token language** - Based on insights from literature
6. **Implement and train** - Compare against baselines
7. **Evaluate comprehensively** - Use all recommended metrics
8. **Analyze and iterate** - Understand what works and why

---

## Citation Information

If using these resources, please cite:

**Papers:**
- Pagnoni et al. (2024) - Byte Latent Transformer
- Schmidt et al. (2024) - Tokenization Is More Than Compression
- Huang et al. (2025) - Over-Tokenized Transformer
- [See papers/README.md for complete list]

**Datasets:**
- Hutter Prize (enwiki8, text8)
- Goyal et al. (2022) - FLORES
- Merity et al. (2016) - WikiText
- Edman et al. (2024) - CUTE
- [See datasets/README.md for complete citations]

**Code:**
- facebook/blt - Byte Latent Transformer
- lucidrains/MEGABYTE-pytorch
- karpathy/minbpe
- google/sentencepiece

---

## Contact and Support

For questions about these resources:
- Check README files in each directory
- Review paper abstracts and documentation
- Consult original paper/dataset/code repositories

---

**Resource gathering phase: COMPLETE ✓**

All necessary papers, datasets, and code have been identified and documented. The experiment runner has everything needed to begin the research phase.
