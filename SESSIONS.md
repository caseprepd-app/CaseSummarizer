# Session Reference

This document maps session numbers referenced in code comments to their decisions.
Sessions represent significant development/design decisions made during pair programming sessions.

## Session Index

| Session | Topic | Key Decision |
|---------|-------|--------------|
| 15 | spaCy Model | Added spaCy download timeouts and thread management |
| 16 | GUI Pagination | Vocabulary display pagination to prevent GUI freezing |
| 20 | Document Chunking | Hierarchical summarization with chunk overlap |
| 25 | ML Boost | Added meta-learner for quality score adjustment |
| 26 | BM25 Corpus | BM25 algorithm with corpus-based IDF scoring |
| 29 | Multi-Corpus | Preprocessing support for multiple corpora |
| 45 | Unified Chunking | Token-based semantic chunking with configurable sizes |
| 47 | Algorithm Weights | Per-algorithm tracking (NER, RAKE, BM25) and time decay |
| 52 | Type Features | Removed unreliable type features, kept only is_person |
| 54 | OCR Confidence | Added source_doc_confidence feature for OCR quality |
| 55 | Two-File Feedback | Split default (shipped) and user feedback files |
| 58 | Rank-Based Scoring | Changed from log-based to rank-based rarity scoring |
| 59 | User Preferences | Made rarity thresholds configurable via GUI |
| 64 | Context Window | Dynamic context window scaling based on GPU VRAM |
| 67 | Fixed Chunk Sizes | Research-based fixed chunk sizes (400-1000 tokens) |
| 68 | Corpus Familiarity | Added corpus familiarity filtering and is_title_case |
| 70 | Name Caching | LRU cache for phrase component scores |
| 76 | Feature Overhaul | Major ML feature overhaul with word-level frequency |
| 77 | Debug Mode Routing | DEBUG_MODE controls feedback routing (temp hard-coded) |
| 78 | TermSources | Per-document confidence tracking with TermSources |
| 79 | Rule-Based Adjustments | Quality score adjustments based on TermSources |
| 80 | Artifact Filter | Common-word variant detection for Person entities |
| 83 | Name Validation | Name dataset validation and domain-specific features |
| 84 | Count Bins | One-hot encoded count bins for ML features |
| 85 | Deduplication | Changed from term-only to (term, count_bin) deduplication |
| 86 | Auto-Train | Auto-train ML model when sufficient data available |
| 130 | Expanded Bins | Added granularity above 7 occurrences + log rarity score |
| 131 | Log Count | Added log-scaled count to preserve magnitude within bins |

## Session Details

### Session 25: ML Boost
Added the meta-learner system for vocabulary quality scoring. The learner trains on user feedback (thumbs up/down) to predict which terms users would approve.

### Session 45: Unified Chunking
Implemented token-based semantic chunking to replace character-based chunking. Uses LangChain's SemanticChunker with gradient breakpoints, enforces token limits post-processing.

### Session 52: Type Features
Removed unreliable type features (is_medical, is_technical, is_place, is_unknown) from ML features. Only `is_person` from NER detection was reliable enough to keep.

### Session 55: Two-File Feedback
Split feedback into two files:
- `config/default_feedback.csv` - Shipped with app (developer training data)
- `%APPDATA%/CasePrepd/feedback/user_feedback.csv` - User's own feedback
User feedback is weighted higher once sufficient samples accumulate.

### Session 68: Corpus Familiarity
Added corpus familiarity scoring - terms that appear in many corpus documents are likely common legal vocabulary and weighted lower. Added `is_title_case` feature.

### Session 76: Feature Overhaul
Major ML feature update:
- Removed: quality_score (circular), num_algorithms (redundant)
- Added: word-level frequency analysis, vowel ratio, medical suffix detection, repeated chars

### Session 78: TermSources
Introduced TermSources class to track which source documents contributed each term occurrence. Enables multi-document confidence features:
- num_source_documents
- doc_diversity_ratio
- mean/median_doc_confidence
- confidence_std_dev
- high_conf_doc_ratio
- all_low_conf

### Session 84: Count Bins
Changed from continuous count feature to one-hot encoded count bins:
- count=1 (possible OCR error)
- count=2-3 (low confidence)
- count=4-6 (moderate)
- count=7+ (reliable)

### Session 130: Expanded Bins
Added finer granularity for high-frequency terms:
- bin_7_20 (mentioned multiple times)
- bin_21_50 (appears throughout document)
- bin_51_plus (major figure in transcript)
Also added `word_log_rarity_score` to distinguish rare vs common words.

## How to Add New Sessions

When making significant decisions, add a new session entry:

1. Use the next available session number
2. Add comment in code: `# Session N: Brief description`
3. Add entry to the table above
4. Optionally add detailed section if decision is complex

Example code comment:
```python
# Session 132: Added XYZ feature for better ABC detection
```
