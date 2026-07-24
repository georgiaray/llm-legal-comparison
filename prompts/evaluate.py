#!/usr/bin/env python3
"""
Classification evaluation script: computes precision, recall, and F1 for a
multi-label classification task against a human-labeled groundtruth set.

WHAT THIS DOES NOT COME WITH -- READ THIS FIRST
=================================================
1. You need your own groundtruth. `data/groundtruth_example.xlsx` ships with
   this repo purely to show the *shape* a groundtruth file can take (columns,
   join key) -- its label columns are empty. There is no shortcut around
   manually labeling a sample of documents yourself before you can measure
   anything. If you don't have that yet, this script has nothing to evaluate.

2. This is NOT a fully objective, plug-and-play metric. Precision/recall
   against a taxonomy assumes there is one right answer per document per
   question, and for many real classification schemes that assumption is
   shaky: a single document can legitimately support more than one label at
   once, two categories in your taxonomy may overlap in scope for a given
   document, and a "disagreement" between the model and your groundtruth can
   reflect a genuine difference of judgment rather than a model error. We
   built this script for a climate-finance-policy classification task with a
   14-17 document groundtruth set, and a manual read-through of a sample of
   the disagreements found that a substantial share were defensible either
   way, not clear model errors. Do not treat the numbers this script prints
   as a final verdict -- read a sample of the actual disagreements (this
   script writes them out precisely so you can do that) before drawing
   conclusions, and report confidence intervals, not bare percentages,
   especially at small sample sizes.

WHAT YOU NEED TO PROVIDE
=========================
1. A taxonomy config (JSON) describing the canonical label set for each
   question you want scored. See `prompts/evaluate_taxonomy_example.json`
   for the format. Two label shapes are supported per question:
     - "flat": labels are bare strings (e.g. a single-dimension category)
     - "paired": labels are [category, value] pairs, combined internally as
       "category: value" (e.g. "Entities: Non-financial corporations") --
       use this when your taxonomy nests values under categories.

2. A groundtruth file (CSV or XLSX) with one row per document: a join-key
   column (default "unique_id") plus one column per question, containing
   the human-assigned label(s) as free text (delimiters like ";" are fine --
   this script fuzzy-matches text fragments against your taxonomy rather
   than requiring exact pre-formatted strings).

3. A predictions file (CSV) in the same shape: join-key column plus one
   column per question, containing the model's raw response text. Both
   freeform text and JSON-structured responses (e.g. {"categories": [...]})
   are supported -- see `parse_and_match()` below.

USAGE
=====
    python prompts/evaluate.py \\
        --groundtruth data/my_groundtruth.csv \\
        --predictions data/my_predictions.csv \\
        --taxonomy prompts/evaluate_taxonomy_example.json \\
        --questions question_2,question_3 \\
        --output data/evaluation_results

Or via the Makefile: `make evaluate GROUNDTRUTH=... PREDICTIONS=... TAXONOMY=...`
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from fuzzywuzzy import fuzz, process
except ImportError:
    print("This script requires fuzzywuzzy (already a project dependency: "
          "`poetry install` or `pip install fuzzywuzzy python-Levenshtein`).",
          file=sys.stderr)
    raise

try:
    from scipy import stats
except ImportError:
    stats = None  # confidence intervals will be skipped with a warning


DEFAULT_MATCH_THRESHOLD = 85  # fuzzywuzzy token_set_ratio, 0-100
HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /\-]{2,60}):\s*(.*)$")
BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s*(.+)$")
NONE_APPLY_RE = re.compile(
    r"none (of the (listed|above))?[^.]*(apply|applicable)"
    r"|not applicable"
    r"|does not (apply|fit|match)"
    r"|doesn.t (apply|fit|match)"
    r"|no (applicable|relevant|matching) (label|categor|subject|instrument)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------

def wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval -- better small-n behaviour than a normal
    approximation. Use this (not a bare percentage) for anything computed
    from a small groundtruth set."""
    if n == 0 or stats is None:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half_width = (z * ((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


# --------------------------------------------------------------------------
# Taxonomy loading
# --------------------------------------------------------------------------

class Taxonomy:
    """Loads a user-supplied taxonomy config and exposes per-question
    canonical label lists, paired/flat mode, and optional aliases/shorthand."""

    def __init__(self, config_path):
        with open(config_path) as f:
            raw = json.load(f)
        self.questions = {}
        self.subject_only_maps = {}
        for q, spec in raw.items():
            paired = spec.get("paired", False)
            if paired:
                pairs = [tuple(p) for p in spec["labels"]]
                labels = [f"{c}: {v}" for c, v in pairs]
                subjects = [v for _, v in pairs]
                if len(subjects) == len(set(s.lower() for s in subjects)):
                    # subject values are unique across categories -- safe to
                    # support a "value alone" fallback match (handles a model
                    # stating the wrong/garbled category but the right value)
                    self.subject_only_maps[q] = {v: f"{c}: {v}" for c, v in pairs}
            else:
                labels = list(spec["labels"])
            self.questions[q] = {
                "paired": paired,
                "labels": labels,
                "aliases": {k.lower(): v for k, v in spec.get("aliases", {}).items()},
                "shorthand": spec.get("shorthand_expansions", {}),
            }

    def labels_for(self, question):
        return self.questions[question]["labels"]

    def is_paired(self, question):
        return self.questions[question]["paired"]

    def alias_for(self, question, candidate):
        return self.questions[question]["aliases"].get(candidate.strip().lower())

    def shorthand_for(self, question, candidate):
        norm = candidate.strip().lower()
        for shorthand, expansion in self.questions[question]["shorthand"].items():
            if shorthand.lower() in norm or norm == shorthand.strip("[]").lower():
                return expansion
        return None

    def subject_only_match(self, question, value_part, threshold):
        subject_map = self.subject_only_maps.get(question)
        if not subject_map or not value_part:
            return None, 0
        best = process.extractOne(value_part, list(subject_map.keys()), scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            return subject_map[best[0]], best[1]
        return None, 0


# --------------------------------------------------------------------------
# Text parsing: raw cell -> candidate label fragments
# --------------------------------------------------------------------------

def _try_parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _walk_json_for_candidates(obj, paired):
    candidates = []

    def walk(node):
        if isinstance(node, dict):
            if paired and "category" in node:
                value = node.get("subject") or node.get("policy") or node.get("value") or node.get("label")
                if value:
                    candidates.append(f"{node['category']}: {value}")
                    return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str) and not paired:
                    candidates.append(item)
                else:
                    walk(item)

    walk(obj)
    return candidates


def _split_into_pieces(text):
    pieces = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        bullet_match = BULLET_LINE_RE.match(line)
        if bullet_match:
            value = bullet_match.group(1).strip()
            if value:
                pieces.append((value, True))
        else:
            for frag in line.split(";"):
                frag = frag.strip()
                if frag:
                    pieces.append((frag, False))
    return pieces


def _extract_freeform_candidates(text, paired):
    text = text.strip()
    if NONE_APPLY_RE.search(text):
        return []  # explicit "no applicable label" -- confident empty prediction

    candidates = []
    marker_match = None
    for marker in ("classification:", "final classification:", "answer:"):
        idx = text.lower().rfind(marker)
        if idx != -1:
            marker_match = text[idx + len(marker):].strip()
    if marker_match is not None and not paired:
        for frag in re.split(r"[;\n]", marker_match):
            frag = frag.strip(" .")
            if frag:
                candidates.append((frag, "classification-marker"))
        if candidates:
            return candidates

    current_header = None
    for piece_text, is_bullet in _split_into_pieces(text):
        if is_bullet:
            value = piece_text.strip(" .")
            if value:
                cand = f"{current_header}: {value}" if (paired and current_header) else value
                candidates.append((cand, "freeform"))
            continue
        header_match = HEADER_LINE_RE.match(piece_text)
        if header_match and len(header_match.group(1).split()) <= 6:
            header, rest = header_match.group(1).strip(), header_match.group(2).strip(" .")
            current_header = header
            if rest:
                cand = f"{header}: {rest}" if paired else rest
                candidates.append((cand, "freeform"))
        else:
            value = piece_text.strip(" .")
            if value:
                cand = f"{current_header}: {value}" if (paired and current_header) else value
                candidates.append((cand, "freeform"))

    if candidates:
        return candidates
    for frag in re.split(r"[;\n]", text):
        frag = frag.strip(" .")
        if frag:
            candidates.append((frag, "freeform"))
    return candidates


def extract_raw_candidates(text, paired):
    if text is None or isinstance(text, float):
        return []
    text = str(text).strip()
    if not text or text.lower() in ("nan", "none"):
        return []
    json_obj = _try_parse_json(text)
    if json_obj is not None:
        return [(c, "json") for c in _walk_json_for_candidates(json_obj, paired)]
    return _extract_freeform_candidates(text, paired)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def match_candidate(candidate, question, taxonomy, threshold):
    alias = taxonomy.alias_for(question, candidate)
    if alias:
        return alias, 100

    labels = taxonomy.labels_for(question)
    best = process.extractOne(candidate, labels, scorer=fuzz.token_set_ratio)
    matched, score = (best[0], best[1]) if best else (None, 0)
    if score >= threshold:
        return matched, score

    if taxonomy.is_paired(question) and ":" in candidate:
        value_part = candidate.rsplit(":", 1)[1].strip()
        sub_matched, sub_score = taxonomy.subject_only_match(question, value_part, threshold)
        if sub_matched:
            return sub_matched, sub_score

    return None, score


def parse_and_match(text, question, taxonomy, threshold=DEFAULT_MATCH_THRESHOLD):
    result = {"matched_labels": set(), "unmatched_fragments": [], "trace": []}
    paired = taxonomy.is_paired(question)
    for candidate, method in extract_raw_candidates(text, paired):
        if taxonomy.is_paired(question):
            expansion = taxonomy.shorthand_for(question, candidate)
            if expansion:
                for lbl in expansion:
                    result["matched_labels"].add(lbl)
                    result["trace"].append({"candidate": candidate, "method": method + "-shorthand",
                                             "matched_label": lbl, "score": 100})
                continue

        matched, score = match_candidate(candidate, question, taxonomy, threshold)
        result["trace"].append({"candidate": candidate, "method": method,
                                 "matched_label": matched, "score": score})
        if matched:
            result["matched_labels"].add(matched)
        else:
            sub_fragments = [p.strip() for p in candidate.split(",") if p.strip()]
            if len(sub_fragments) > 1:
                any_matched = False
                for sub in sub_fragments:
                    sub_matched, sub_score = match_candidate(sub, question, taxonomy, threshold)
                    result["trace"].append({"candidate": sub, "method": method + "-comma-split",
                                             "matched_label": sub_matched, "score": sub_score})
                    if sub_matched:
                        result["matched_labels"].add(sub_matched)
                        any_matched = True
                if not any_matched:
                    result["unmatched_fragments"].append(candidate)
            else:
                result["unmatched_fragments"].append(candidate)
    return result


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return precision, recall, f1


def load_table(path, join_key):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if join_key not in df.columns:
        raise SystemExit(f"Join key '{join_key}' not found in {path}. Columns: {list(df.columns)}")
    return df


def run_evaluation(groundtruth_df, predictions_df, questions, taxonomy, join_key, threshold):
    gt = groundtruth_df.set_index(join_key)
    pred = predictions_df.set_index(join_key)
    common_ids = gt.index.intersection(pred.index)
    if len(common_ids) == 0:
        raise SystemExit(
            f"No overlapping '{join_key}' values between groundtruth and predictions. "
            "Nothing to evaluate -- check the join key matches in both files."
        )
    if len(common_ids) < len(gt.index) or len(common_ids) < len(pred.index):
        print(f"WARNING: only {len(common_ids)} of {len(gt.index)} groundtruth rows and "
              f"{len(pred.index)} prediction rows share a '{join_key}' -- scoring the overlap only.",
              file=sys.stderr)

    per_doc_rows, trace_rows = [], []
    for uid in common_ids:
        for q in questions:
            if q not in gt.columns or q not in pred.columns:
                continue
            gt_result = parse_and_match(gt.loc[uid, q], q, taxonomy, threshold)
            pred_result = parse_and_match(pred.loc[uid, q], q, taxonomy, threshold)
            gt_set, pred_set = gt_result["matched_labels"], pred_result["matched_labels"]
            tp, fp, fn = len(gt_set & pred_set), len(pred_set - gt_set), len(gt_set - pred_set)
            per_doc_rows.append({
                join_key: uid, "question": q,
                "gt_labels": "; ".join(sorted(gt_set)), "pred_labels": "; ".join(sorted(pred_set)),
                "tp": tp, "fp": fp, "fn": fn,
                "gt_unmatched": "; ".join(gt_result["unmatched_fragments"]),
                "pred_unmatched": "; ".join(pred_result["unmatched_fragments"]),
            })
            for t in gt_result["trace"]:
                trace_rows.append({join_key: uid, "question": q, "source": "groundtruth", **t})
            for t in pred_result["trace"]:
                trace_rows.append({join_key: uid, "question": q, "source": "prediction", **t})

    per_doc_df = pd.DataFrame(per_doc_rows)
    trace_df = pd.DataFrame(trace_rows)

    summary_rows = []
    for q, grp in per_doc_df.groupby("question"):
        tp, fp, fn = grp.tp.sum(), grp.fp.sum(), grp.fn.sum()
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        p_lo, p_hi = wilson_ci(tp, tp + fp)
        r_lo, r_hi = wilson_ci(tp, tp + fn)
        summary_rows.append({
            "question": q, "n_docs": len(grp), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "precision_wilson_lo": p_lo, "precision_wilson_hi": p_hi,
            "recall": recall, "recall_wilson_lo": r_lo, "recall_wilson_hi": r_hi,
            "f1": f1,
        })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, per_doc_df, trace_df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SUBJECTIVITY_WARNING = """
============================================================================
REMINDER: this metric is not fully objective.

Before reporting the numbers below, read a sample of the actual
disagreements in {output_dir}/per_document.csv (rows with fp > 0 or fn > 0)
against your source documents. In our own use of this script, roughly half
of the "incorrect" cases we manually checked turned out to be genuine
taxonomy ambiguity or a defensible alternative reading, not a model error --
treat the precision/recall figures as a conservative floor, not a verdict,
until you've done that check yourself.
============================================================================
"""


def main():
    parser = argparse.ArgumentParser(
        description="Score model classification predictions against a human-labeled groundtruth set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--groundtruth", required=True, help="Path to groundtruth CSV/XLSX (you must provide this).")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV (you must provide this).")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON config (see evaluate_taxonomy_example.json).")
    parser.add_argument("--questions", required=True, help="Comma-separated list of question/column names to score.")
    parser.add_argument("--join-key", default="unique_id", help="Column used to join groundtruth and predictions (default: unique_id).")
    parser.add_argument("--threshold", type=int, default=DEFAULT_MATCH_THRESHOLD, help=f"Fuzzy match threshold, 0-100 (default: {DEFAULT_MATCH_THRESHOLD}).")
    parser.add_argument("--output", default="data/evaluation_results", help="Output directory (default: data/evaluation_results).")
    args = parser.parse_args()

    questions = [q.strip() for q in args.questions.split(",") if q.strip()]
    taxonomy = Taxonomy(args.taxonomy)

    gt_df = load_table(args.groundtruth, args.join_key)
    pred_df = load_table(args.predictions, args.join_key)

    summary_df, per_doc_df, trace_df = run_evaluation(
        gt_df, pred_df, questions, taxonomy, args.join_key, args.threshold
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary_precision_recall_f1.csv", index=False)
    per_doc_df.to_csv(out_dir / "per_document.csv", index=False)
    trace_df.to_csv(out_dir / "parse_trace.csv", index=False)

    unmatched_rate = trace_df.loc[trace_df.source == "prediction", "matched_label"].isna().mean()
    pd.set_option("display.width", 140)
    print(summary_df.round(3).to_string(index=False))
    print(f"\nPrediction-side fragment unmatched rate: {unmatched_rate:.1%} "
          f"(low unmatched rate = taxonomy/parser is capturing the model's output faithfully; "
          f"a high rate means check parse_trace.csv before trusting the scores above)")
    print(SUBJECTIVITY_WARNING.format(output_dir=args.output))
    print(f"Wrote: {out_dir}/summary_precision_recall_f1.csv, per_document.csv, parse_trace.csv")


if __name__ == "__main__":
    main()
