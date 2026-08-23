# Licensing of the released data

## Summary

| What ships | Terms |
|---|---|
| `cosda/`, `scripts/`, `configs/`, `reproduce.sh` | Apache-2.0 (see `LICENSE`) |
| Generated candidates from **news** cells (`*/news_topic_classification/*`) | CC BY-NC 4.0 (inherited, see below) |
| Generated candidates from **sentiment** cells (`*/sentiment_classification/*`) | CC BY 4.0 (inherited) |
| Audit scores, selector decisions, all files under `results/` | CC BY 4.0 |
| Source-corpus text | **not redistributed** — see `README.md` |

## Why the generated candidates carry inherited terms

Every candidate was generated conditioned on gold examples from a source dataset, and is
therefore plausibly "Adapted Material" under those datasets' Creative Commons licences.
Whether prompt-conditioned model output is legally adapted material is untested, so we
take the conservative reading.

- **MasakhaNEWS** (news cells). The dataset card states the data is CC BY-NC 4.0
  (<https://huggingface.co/datasets/masakhane/masakhanews>), and Adelani et al. (2023)
  state release "under academic license or CC BY-NC 4.0". CC BY-NC 4.0 §2(a)(1)(B)
  permits sharing Adapted Material for NonCommercial purposes only, so the news-derived
  candidates are marked NC. Note the repository's YAML tag reads `afl-3.0`, which
  contradicts the card prose; we follow the more restrictive reading.
- **AfriSenti** (sentiment cells). The authors' own repositories state CC BY 4.0
  (<https://github.com/afrisenti-semeval/afrisent-semeval-2023>, and the lead author's
  <https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment>). No NC or
  ShareAlike clause propagates. The `masakhane/afrisenti` mirror carries a
  `cc-by-nc-sa-2.0` YAML tag that conflicts with all three author-controlled sources and
  that we believe is a metadata error; we flag it here rather than silently resolve it.

## Rights we cannot grant

Neither dataset licence reaches the upstream content, and neither do we:

- MasakhaNEWS reproduces article text from BBC, VOA and other outlets. Those bodies
  remain the publishers' copyright, and the CC grant covers only the annotations and the
  compilation. This is why we withhold the source text rather than redistribute it.
- AfriSenti reproduces tweet text. X's developer terms generally permit redistributing
  tweet identifiers rather than hydrated text, independently of the CC grant, and tweets
  are user-generated content from identifiable people.

If you rehydrate the gold set with `scripts/rehydrate_gold.py`, you obtain that content
directly from the source datasets and under their terms, not from us.

## Attribution

- Adelani et al. 2023, *MasakhaNEWS*, IJCNLP-AACL. <https://aclanthology.org/2023.ijcnlp-main.10/>
- Muhammad et al. 2023, *AfriSenti*, EMNLP. <https://aclanthology.org/2023.emnlp-main.862/>
