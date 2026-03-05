"""
Deprecated vocabulary extraction algorithms.

These algorithms have been moved here because they are too slow for large
documents without a dedicated GPU:

- KeyBERT: Generates all 1-3 gram candidates, encodes each through a
  transformer, then runs O(n²) MMR. Hangs/takes 20+ minutes on 177-page docs.

- GLiNER: 209MB model, chunks text into ~300-word pieces. Heavy and overlaps
  significantly with the faster algorithms (NER, RAKE, YAKE, TopicRank).

The implementations are preserved here for reference or future re-enablement
if GPU support becomes available.
"""
