# Resources Catalog

**Research Project**: An Artificial Token Language for More Efficient LLMs
**Date**: December 7, 2025
**Phase**: Resource Gathering Complete

---

## Summary

This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories. All materials are organized and ready for the experiment runner phase.

### Resource Summary
- **Papers Downloaded**: 20 (16 from 2024-2025)
- **Datasets Identified**: 5 (1 downloaded, 4 with download scripts)
- **Code Repositories Cloned**: 5
- **Documentation**: 4 comprehensive markdown files
- **Total Disk Usage**: ~30 MB (papers + code, excluding large datasets)

---

## Papers

### Overview
Total papers downloaded: **20**

All papers are stored in `papers/` directory as PDFs with descriptive filenames.

### Papers by Category

#### Byte-Level and Tokenizer-Free Models (6 papers)
| Title | File | Year | Code |
|-------|------|------|------|
| Byte Latent Transformer: Patches Scale Better Than Tokens | 2412.09871_byte_latent_transformer.pdf | 2024 | ✅ Yes |
| MEGABYTE: Predicting Million-byte Sequences | 2305.07185_megabyte.pdf | 2023 | ⚠️ Community |
| ByT5: Towards a token-free future | 2105.13626_byt5.pdf | 2021 | ✅ Yes |
| T-FREE: Tokenizer-Free Generative LLMs | 2406.19223_tfree.pdf | 2024 | ❌ No |
| SpaceByte: Towards Deleting Tokenization | 2404.14408_spacebyte.pdf | 2024 | ❌ No |
| MrT5: Dynamic Token Merging | 2410.20771_mrt5.pdf | 2024 | ❌ No |

#### Vocabulary Compression and Optimization (4 papers)
| Title | File | Year | Code |
|-------|------|------|------|
| LLM Vocabulary Compression for Low-Compute | 2411.06371_vocab_compression.pdf | 2024 | ❌ No |
| Fast Vocabulary Transfer | 2402.09977_fast_vocab_transfer.pdf | 2024 | ❌ No |
| Vocabulary Trimming | 2305.15020_vocab_trimming.pdf | 2023 | ❌ No |
| Over-Tokenized Transformer | 2501.16975_overtokenized_transformer.pdf | 2025 | ❌ No |

#### Alternative Tokenization Methods (5 papers)
| Title | File | Year | Code |
|-------|------|------|------|
| BPE is Suboptimal | 2004.03720_bpe_suboptimal.pdf | 2020 | ❌ No |
| Charformer: Gradient-based Tokenization | 2106.12672_charformer.pdf | 2021 | ❌ No |
| Rethinking Tokenization | 2403.00417_rethinking_tokenization.pdf | 2024 | ❌ No |
| ReTok: Efficient Tokenizer | 2410.04335_retok_efficient_tokenizer.pdf | 2024 | ❌ No |
| Hierarchical Transformers | 2501.10322_hierarchical_transformers.pdf | 2025 | ❌ No |

#### Theoretical Foundations (5 papers)
| Title | File | Year | Code |
|-------|------|------|------|
| Toward a Theory of Tokenization | 2404.08335_theory_of_tokenization.pdf | 2024 | ❌ No |
| Unpacking Tokenization | 2403.06265_unpacking_tokenization.pdf | 2024 | ❌ No |
| Tokenization More Than Compression | 2402.18376_tokenization_more_than_compression.pdf | 2024 | ❌ No |
| From Tokens to Characters | 2412.03719_tokens_to_characters.pdf | 2024 | ❌ No |
| Neurally Compressed Text | 2404.03626_neurally_compressed_text.pdf | 2024 | ❌ No |

### Key Papers (Must Read First)
1. **BLT** (2412.09871) - SOTA byte-level, breakthrough results ⭐⭐⭐⭐⭐
2. **Unpacking Tokenization** (2403.06265) - Compression-quality correlation ⭐⭐⭐⭐⭐
3. **Theory of Tokenization** (2404.08335) - Theoretical foundations ⭐⭐⭐⭐
4. **Vocab Compression** (2411.06371) - Practical efficiency gains, uses TinyStories ⭐⭐⭐⭐
5. **ByT5** (2105.13626) - Production baseline ⭐⭐⭐⭐

### Detailed Documentation
See `papers/README.md` for:
- Detailed descriptions of each paper
- Authors and arXiv links
- Why each paper is relevant
- Key themes and insights
- Recommended baselines and metrics

---

## Datasets

### Overview
Total datasets identified: **5** (1 downloaded, 4 ready to download)

All datasets have comprehensive download instructions in `datasets/README.md`.

### Dataset Summary Table

| Dataset | Size | Status | Primary Use | Priority |
|---------|------|--------|-------------|----------|
| enwik8 | 100 MB | ✅ **Downloaded** | Compression benchmark, byte-level LM | High |
| TinyStories | ~500 MB | 📥 Ready | Small-scale LM training, controlled vocab | **Critical** |
| WikiText-103 | ~500 MB | 📥 Ready | Standard LM benchmark | **Critical** |
| C4 | ~750 GB | 📥 Ready (streaming) | Large-scale pretraining | Medium |
| GLUE/SuperGLUE | ~1-5 GB | 📥 Ready | Downstream evaluation | Low |

### Downloaded Datasets

#### enwik8 ✅
- **Location**: `datasets/enwik8/enwik8`
- **Size**: 100,000,000 bytes (uncompressed)
- **Format**: UTF-8 XML (Wikipedia)
- **Splits**: Standard: 90M train / 5M valid / 5M test
- **Status**: Ready to use
- **Used in**: BLT, ByT5, MEGABYTE papers

### Datasets Ready to Download

#### TinyStories 📥
- **Source**: HuggingFace `roneneldan/TinyStories`
- **Download script**: In `datasets/README.md`
- **Recommended for**: Initial experiments, fast iteration
- **Used in**: Vocabulary Compression paper (2411.06371)

#### WikiText-103 📥
- **Source**: HuggingFace `Salesforce/wikitext`
- **Download script**: In `datasets/README.md`
- **Recommended for**: Standard benchmark reporting
- **Used in**: Most tokenization papers

#### C4 📥
- **Source**: HuggingFace `allenai/c4`
- **Download**: **Streaming recommended** (750GB full download)
- **Recommended for**: Large-scale validation
- **Used in**: T5, ByT5, many LLM papers

#### GLUE/SuperGLUE 📥
- **Source**: HuggingFace `nyu-mll/glue`, `super_glue`
- **Download scripts**: In `datasets/README.md`
- **Recommended for**: Downstream task evaluation
- **Used in**: Standard NLP benchmarking

### Storage Requirements
- **Minimal setup** (enwik8 + TinyStories): ~600 MB ✅
- **Standard setup** (+ WikiText-103): ~1.1 GB
- **Full setup** (+ C4 sample + GLUE): ~5-10 GB
- **Production setup** (+ C4 full): ~800 GB

### Quick Start
```bash
cd datasets
python download_datasets.py  # Script in README.md
```

### Detailed Documentation
See `datasets/README.md` for:
- Complete download instructions for each dataset
- Loading examples (HuggingFace datasets library)
- Sample data from each dataset
- Git-friendly setup (.gitignore configured)
- Storage optimization tips

---

## Code Repositories

### Overview
Total repositories cloned: **5**

All repositories are in `code/` directory with full git history.

### Repository Summary Table

| Repository | Organization | Purpose | Language | Stars | Relevance |
|------------|--------------|---------|----------|-------|-----------|
| BLT | Meta FAIR | SOTA byte-level LLM | PyTorch | New (Dec 2024) | ⭐⭐⭐⭐⭐ |
| MEGABYTE | Community (shjwudp) | Multi-scale byte modeling | PyTorch | ~200 | ⭐⭐⭐⭐ |
| ByT5 | Google Research | Production byte-level T5 | JAX/T5X | ~500 | ⭐⭐⭐⭐⭐ |
| SentencePiece | Google | Subword tokenization | C++/Python | ~9K | ⭐⭐⭐⭐ |
| HuggingFace Tokenizers | HuggingFace | Fast tokenization library | Rust/Python | ~8K | ⭐⭐⭐⭐⭐ |

### Repository Details

#### 1. BLT (Byte Latent Transformer) ⭐⭐⭐⭐⭐
- **Location**: `code/blt/`
- **URL**: https://github.com/facebookresearch/blt
- **Purpose**: State-of-the-art byte-level language model
- **Pre-trained Models**: Yes (up to 8B parameters)
- **Key Features**:
  - Dynamic entropy-based patching
  - Matches token-based LLM performance at scale
  - Inference efficiency through adaptive patches
- **Usage**: Baseline for byte-level approaches, reference implementation for dynamic tokenization

#### 2. MEGABYTE ⭐⭐⭐⭐
- **Location**: `code/megabyte/`
- **URL**: https://github.com/shjwudp/megabyte
- **Purpose**: Hierarchical multi-scale transformer
- **Pre-trained Models**: No (community implementation)
- **Key Features**:
  - Local + global model architecture
  - Sub-quadratic attention
  - Tokenization-free
- **Usage**: Understanding hierarchical approaches, architectural inspiration

#### 3. ByT5 ⭐⭐⭐⭐⭐
- **Location**: `code/byt5/`
- **URL**: https://github.com/google-research/byt5
- **Purpose**: Production-ready byte-level T5
- **Pre-trained Models**: Yes (small, base, large, xl, xxl via HuggingFace)
- **Key Features**:
  - Official Google implementation
  - Pre-trained on mC4
  - Compatible with HuggingFace Transformers
- **Usage**: Strong baseline, production reference

#### 4. SentencePiece ⭐⭐⭐⭐
- **Location**: `code/sentencepiece/`
- **URL**: https://github.com/google/sentencepiece
- **Purpose**: Train BPE and Unigram tokenizers
- **Key Features**:
  - Fast C++ implementation
  - Supports BPE, Unigram LM, character, word
  - Language-independent
- **Usage**: Training baseline tokenizers, comparison against standard methods

#### 5. HuggingFace Tokenizers ⭐⭐⭐⭐⭐
- **Location**: `code/tokenizers/`
- **URL**: https://github.com/huggingface/tokenizers
- **Purpose**: Fast tokenization library for custom implementations
- **Key Features**:
  - Rust implementation (extremely fast)
  - Extensible for custom algorithms
  - Compatible with Transformers library
- **Usage**: Implementing custom artificial token language

### Installation Status
- All repositories cloned ✅
- Dependencies NOT installed yet (experiment runner will handle)
- Full git history preserved for reference

### Detailed Documentation
See `code/README.md` for:
- Detailed descriptions of each repository
- Key files and entry points
- Usage examples
- Installation instructions
- Relevance ratings and use cases

---

## Documentation Files

### 1. literature_review.md (THIS IS THE MAIN DELIVERABLE)
**Contents**:
- Executive summary of 20 papers
- Detailed analysis of each paper
- Common methodologies and metrics
- Standard baselines and benchmarks
- Gaps and opportunities
- Detailed recommendations for experiments

**Key Sections**:
- Theoretical foundations (4 papers)
- Byte-level models (6 papers)
- Vocabulary compression (4 papers)
- Alternative tokenization (5 papers)
- Compression-performance correlation (3 papers)

**Recommendations**:
- Datasets: TinyStories → enwik8 → WikiText-103 → C4
- Baselines: BPE, Unigram LM, ByT5, BLT
- Metrics: Perplexity, compression ratio, throughput, memory
- Experimental pipeline: 4-phase approach over 10-12 weeks

### 2. papers/README.md
**Contents**:
- Complete catalog of 20 papers
- Authors, years, arXiv IDs, file sizes
- Detailed relevance explanations
- Key themes and trends
- Research implications

### 3. datasets/README.md
**Contents**:
- Comprehensive dataset documentation
- Download instructions for each dataset
- Loading examples (Python code)
- Sample data from each dataset
- Storage requirements and optimization
- Quick start scripts

### 4. code/README.md
**Contents**:
- Complete catalog of 5 repositories
- Purpose and key features
- Installation instructions
- Usage examples
- Relevance ratings
- Recommended usage priority

### 5. resources.md (THIS FILE)
**Contents**:
- High-level summary of all resources
- Quick reference tables
- Status tracking
- Next steps for experiment runner

---

## Resource Gathering Process

### Search Strategy
**Keywords Used**:
- "artificial token language" LLM
- "compact tokenization" language models
- "token compression" efficiency
- "byte-level" transformers
- "vocabulary optimization"
- "learned tokenization"

**Sources Searched**:
- arXiv (primary source)
- Semantic Scholar
- Papers with Code
- Google Scholar (secondary)
- GitHub (for code repositories)

### Selection Criteria

#### Papers
✅ **Included if**:
- Directly related to tokenization efficiency or alternatives
- Recent (2020-2025, preference for 2024-2025)
- High quality (published or strong preprints)
- Provides code, baselines, or methodological insights

❌ **Excluded if**:
- Too tangential to core research question
- Purely theoretical without practical implications
- Duplicates existing coverage
- Poor quality or unclear contributions

#### Datasets
✅ **Included if**:
- Used in multiple tokenization papers
- Standard benchmark in the field
- Suitable for testing hypothesis
- Reasonable size for experimentation

#### Code
✅ **Included if**:
- Official or high-quality implementation
- Useful for baselines or reference
- Actively maintained
- Compatible with standard frameworks (PyTorch/JAX)

---

## Challenges Encountered

### Challenge 1: Dataset Size Management
- **Issue**: C4 is 750GB, too large to download fully
- **Solution**: Documented streaming approach, validation set download, 1% sampling
- **Status**: Resolved ✅

### Challenge 2: Multiple BLT Paper Versions
- **Issue**: Found references to BLT but needed to verify latest version
- **Solution**: Confirmed arXiv 2412.09871 (Dec 2024) is latest, code on GitHub
- **Status**: Resolved ✅

### Challenge 3: Some Papers Without Code
- **Issue**: 12 out of 20 papers don't have public code
- **Solution**: Focused on papers with code for baselines, others for methodological insights
- **Status**: Acceptable - 5 code repos provide sufficient baseline coverage ✅

### Challenge 4: Git-Friendly Dataset Storage
- **Issue**: Datasets too large for git, but need to document
- **Solution**: Created .gitignore for datasets, comprehensive download instructions
- **Status**: Resolved ✅

---

## Gaps and Workarounds

### Gap 1: No Official MEGABYTE Implementation
- **Gap**: Meta's MEGABYTE paper has no official code release
- **Workaround**: Using community implementation (shjwudp/megabyte)
- **Impact**: Minor - can still understand architecture and approach

### Gap 2: Some Recent Papers (2025) May Not Be Final
- **Gap**: Papers from Jan 2025 might be preprints
- **Workaround**: Verified on arXiv, treat as latest available research
- **Impact**: Minimal - used for insights, not critical baselines

### Gap 3: No Pre-existing Artificial Language Datasets
- **Gap**: No datasets of text in artificial token languages (expected - novel research)
- **Workaround**: Will create as part of experiment
- **Impact**: None - this is the research contribution

---

## Recommendations for Experiment Design

### 1. Primary Dataset: TinyStories
**Why**:
- Used in vocabulary compression paper (validates our approach)
- Small, controlled vocabulary
- Fast iteration
- Sufficient for proof-of-concept

**How to use**:
- Train small LLM (125M-1B params)
- Design artificial token language optimized for this data
- Compare against BPE, Unigram, ByT5 baselines

### 2. Primary Baseline: ByT5
**Why**:
- Pre-trained models available (no training cost)
- Production-ready implementation
- Strong performance in literature
- Direct comparison: artificial language vs. raw bytes

**How to use**:
- Load ByT5-small from HuggingFace
- Fine-tune on TinyStories (or evaluate zero-shot)
- Compare perplexity, compression, efficiency

### 3. Secondary Baseline: BPE (via SentencePiece)
**Why**:
- Industry standard
- Easy to train on TinyStories
- Strong reference point

**How to use**:
- Train BPE tokenizer on TinyStories (vocab size 5K, 10K, 20K)
- Train LLM with same architecture as artificial language model
- Direct comparison: artificial language vs. standard BPE

### 4. Primary Metrics
1. **Perplexity**: Quality of language modeling
2. **Compression ratio**: Tokens per byte or tokens per word
3. **Throughput**: Tokens/second during inference
4. **Memory**: Parameter count, activation memory

### 5. Success Criteria (from Literature)
- ✅ **3-5x compression** vs. BPE (based on vocab compression papers)
- ✅ **Match or beat perplexity** of BPE baseline
- ✅ **Competitive with ByT5** on language modeling
- ✅ **Faster inference** than ByT5 (fewer tokens than bytes)
- ✅ **Validate compression-quality correlation** (Unpacking Tokenization paper)

---

## Next Steps for Experiment Runner

### Immediate Actions (Week 1)
1. ✅ Review `literature_review.md` - understand research landscape
2. ✅ Review `papers/README.md` - identify key papers to read
3. 📥 Download TinyStories dataset (primary experiment data)
4. 📥 Download WikiText-103 (secondary benchmark)
5. 🛠️ Set up Python environment with PyTorch, HuggingFace libraries
6. 🛠️ Install HuggingFace Tokenizers for custom implementation

### Week 2-3: Design Phase
1. Analyze TinyStories vocabulary and patterns
2. Design artificial token language using principles from BLT, Vocab Compression
3. Implement in HuggingFace Tokenizers
4. Train baseline tokenizers (BPE, Unigram) on TinyStories

### Week 4-6: Training Phase
1. Train small LLM (125M-1B params) with artificial language
2. Train identical LLM with BPE baseline
3. Load ByT5-small for comparison
4. Measure perplexity, compression, efficiency

### Week 7-9: Validation Phase
1. Test on enwik8 compression benchmark
2. Test on WikiText-103 language modeling
3. If successful: downstream tasks (GLUE subset)
4. Analyze results, iterate on design

### Week 10-12: Scaling (if successful)
1. Train larger model (1B+ params) on C4 validation
2. Full benchmark suite
3. Prepare results for publication

---

## Resource Completeness Checklist

### Papers ✅
- [x] Searched arXiv, Semantic Scholar, Papers with Code
- [x] Downloaded 20 relevant papers
- [x] Organized in `papers/` directory
- [x] Created comprehensive README with descriptions
- [x] Identified key papers and themes

### Datasets ✅
- [x] Identified 5 suitable datasets
- [x] Downloaded enwik8 (100MB)
- [x] Created download instructions for others
- [x] Configured .gitignore for git-friendly storage
- [x] Created comprehensive README with examples

### Code ✅
- [x] Searched GitHub for relevant repositories
- [x] Cloned 5 key repositories
- [x] Created comprehensive README
- [x] Documented installation and usage
- [x] Identified baselines and tools

### Documentation ✅
- [x] Created literature_review.md (main deliverable)
- [x] Created papers/README.md (paper catalog)
- [x] Created datasets/README.md (dataset guide)
- [x] Created code/README.md (code guide)
- [x] Created resources.md (this file)

### Analysis ✅
- [x] Synthesized findings across papers
- [x] Identified gaps and opportunities
- [x] Formulated recommendations
- [x] Proposed experimental pipeline
- [x] Defined success criteria

---

## Final Summary

### Resources Gathered
- **20 papers** covering tokenization, byte-level models, vocabulary compression, and theoretical foundations
- **5 datasets** (1 downloaded, 4 ready): TinyStories, WikiText-103, enwik8, C4, GLUE
- **5 code repositories**: BLT, MEGABYTE, ByT5, SentencePiece, HuggingFace Tokenizers
- **5 documentation files**: Comprehensive guides for papers, datasets, code, literature review, resources

### Key Insights
1. **Strong support for hypothesis**: Multiple papers show efficiency gains from optimized tokenization
2. **Clear opportunity**: No work designs artificial language from first principles
3. **Validated approach**: Compression correlates with quality, byte-level models work at scale
4. **Practical path forward**: TinyStories → BPE/ByT5 baselines → 3-5x compression target

### Ready for Experiment Runner
All resources are organized, documented, and ready for the experiment runner to begin implementation:
- Literature review provides theoretical foundation and guidance
- Datasets are identified with download scripts
- Code repositories provide baselines and implementation tools
- Comprehensive documentation enables independent work

**Total Time Spent**: ~3 hours (within recommended 2.5-3.5 hour budget)
**Completion Date**: December 7, 2025
**Status**: ✅ **RESOURCE GATHERING COMPLETE**

---

## Contact Information for Resources

### Papers
- **arXiv**: https://arxiv.org (for latest versions)
- **Papers with Code**: https://paperswithcode.com (for implementations)

### Datasets
- **HuggingFace Datasets**: https://huggingface.co/datasets
- **Matt Mahoney's Page**: http://mattmahoney.net/dc/textdata.html (enwik8/9)

### Code
- **GitHub**: All repositories cloned locally in `code/`
- **HuggingFace Models**: https://huggingface.co/models (pre-trained ByT5, BLT)

### Questions or Issues
Refer to the comprehensive READMEs in each subdirectory for detailed information and troubleshooting.

---

**END OF RESOURCES CATALOG**
