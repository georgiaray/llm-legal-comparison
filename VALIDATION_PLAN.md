# Validation Plan — Instructions

Context: this project (VectorLaw) has a paper under review. The paper overclaims two things: (1) "precision and recall... baked into VectorLaw" for the extraction phase — this doesn't exist in code, only a manual audit did; (2) a two-run consistency check for classification, presented as validation but actually only shows reliability, not accuracy. A reviewer flagged validation as "thin." No additional human labeling is available. Do NOT add a second/cross-model LLM judge — that option was considered and rejected.

Do these four things.

## 1. Real precision/recall for classification (highest priority)

Existing human-labeled groundtruth (14–17 docs) already exists — no new labeling needed, only new analysis code.

Data sources (all in `law_comparisons_archive/`):
- Groundtruth: `phase_2/old_results/groundtruth_df.csv`, `phase_2/results/groundtruth_df_run_2.csv` (columns `question_2`–`question_5`, keyed by `unique_id`)
- Model responses to score against it: `phase_2/old_results/merged.csv`, `all_responses.csv`, `larger_model_no_review.csv`, `larger_model_with_review.csv`, `larger_model_with_review_run_2.csv`, `smaller_model_no_review.csv`, `smaller_model_with_review.csv`; and `phase_2/results/` equivalents (`all_responses_run_2.csv`, `large_model_with_review_run_2.csv`, `larger_model_no_review_run_2.csv`, `larger_model_with_review_run_2.csv`, `small_model_no_review_run_2.csv`, `small_model_with_review_run_2.csv`)
- Existing LLM-judge scores for comparison: `phase_2/data/evaluation_results_*.json` (8 files — 2 models × review on/off × 2 runs)

Steps:
- Normalize groundtruth label text per question: unify delimiters (`;` vs `,` both used), fix casing, strip trailing punctuation, fix known typos (e.g. "Entuty" → "Entity"). Enumerate the resulting canonical label set per question.
- Map each model response onto that same label set. Use fuzzy matching (`fuzzywuzzy`, already a dependency) with a confidence threshold. Anything below threshold: do not force a match — log it as unmatched/ambiguous rather than guessing, and report the unmatched rate.
- Spot-check a sample of the parser's output by hand against the raw text before trusting any resulting numbers. This step matters — a bad parser produces confident-looking but wrong numbers.
- Compute multi-label precision/recall/F1 (per-category and macro-averaged) — `sklearn.metrics.precision_recall_fscore_support` with `MultiLabelBinarizer`, or manual set math. Do this per run (all 8 configs), so the existing large-vs-small-model / review-vs-no-review comparison in the paper can be reported with real metrics instead of/alongside the LLM-judge score.
- Deliverable: a table of precision/recall/F1 per question per run, plus a short writeup of parser accuracy/limitations (unmatched rate, any labels that don't cleanly map).

## 2. Confidence intervals on all accuracy figures

Apply to both the existing LLM-judge accuracy numbers (in the `evaluation_results_*.json` files) and the new precision/recall numbers from step 1. Sample size is small (n=14–17) — use Wilson or Clopper-Pearson binomial confidence intervals, not bare percentages. Report these alongside every accuracy/pass-rate figure that goes in the paper.

## 3. Fix the extraction-phase claim in the paper

Current text claims precision/recall "are baked into VectorLaw" for extraction. This is not true — nothing in the codebase computes this. What actually happened: manual cross-referencing of the Climate Policy Radar database against "Climate Laws of the World," documented in `law_comparisons_archive/CPR_database_errors.docx` (duplicates, broken links, missing documents found per jurisdiction).

- Read that doc, quantify what it shows (e.g. count of duplicate/broken-link/missing issues found per jurisdiction, and successful text-extraction rate — `utils/scraping.py` and `utils/extract.py` already report success/fail counts).
- Replace the "precision and recall... baked into VectorLaw" paragraph with an accurate description: manual completeness audit cross-referencing two independent databases, plus the extraction success rate the pipeline already reports. Do not use the terms "precision" or "recall" for this phase unless a true relevant/irrelevant universe is established (it isn't currently).

## 4. Reframe the classification reliability claim + add validity/reliability distinction

- The two-run consistency result (same pass count, <3% score variation) shows measurement *reliability*, not *accuracy*. Relabel it explicitly as a reliability check in the text.
- Add a short paragraph distinguishing three separate questions the paper currently conflates: (a) measurement reliability — consistent across repeated runs — covered by the existing two-run check; (b) classification accuracy against ground truth — covered by the new precision/recall from step 1; (c) criteria/schema validity — whether the classification questions themselves are the right ones — the paper already correctly argues this can't be statistically validated and rests on reasoned justification tied to the literature (existing paragraph, keep it, but move it earlier/make it more prominent given the reviewer's comment).

## 5. Add a limitations statement on the framework-level validation gap

The reviewer's broader complaint — no evaluation of whether the tool actually delivers useful comparative insight for a real user — cannot be answered with the current resource constraints (no human/expert reviewers available). Do not imply this is covered by the other work above. Add an explicit limitations sentence: current validation is internal/automated only; independent expert evaluation of the tool's comparative outputs is out of scope for this study and is flagged as necessary future work.

## Open items to confirm with Georgia before starting

- Where should the new precision/recall analysis code live — in `law_comparisons_archive/` (project-specific, matches where the source data already is) or generalized into `law_comparisons/prompts/` as a reusable script? Leaning toward the former since this is paper-specific data, but confirm.
- Location of the actual paper draft (not found in either mounted folder) — needed to make the actual text edits from steps 3–5.
