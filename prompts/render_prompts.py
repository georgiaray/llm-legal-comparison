#!/usr/bin/env python3
"""
Render Prompts to Markdown

Generates PROMPTS.md from the actual prompt source in prompts_example.py and
judge_prompts.py, so the readable doc can never drift out of sync with the
prompts that are actually used. Run this again and commit the result whenever
either source file changes.

Usage:
    python prompts/render_prompts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import prompts_example as p
from prompts import judge_prompts as j

# Placeholder markers substituted in for the runtime-only arguments
# (document text, retrieved chunks, few-shot examples, etc.) so the rendered
# prompt shows the exact fixed instructions/schema a document actually saw,
# with only the per-document parts clearly marked as placeholders.
DOC_TEXT = "<DOCUMENT_TEXT>"
INSTRUMENT_SUMMARY = "<INSTRUMENT_SUMMARY>"
RELEVANT_CHUNKS = "<RELEVANT_CHUNKS_FROM_VECTOR_STORE>"
EXAMPLES = "<FEW_SHOT_EXAMPLES>"


def code_block(text, lang=""):
    return f"```{lang}\n{text.strip()}\n```\n"


def render():
    lines = []
    lines.append("# Prompts Used in This Study")
    lines.append("")
    lines.append(
        "This is a rendered, human-readable view of the exact prompts used in the "
        "published study this repository accompanies (climate finance policy "
        "classification). It is generated directly from "
        "[`prompts_example.py`](prompts_example.py) and [`judge_prompts.py`](judge_prompts.py) "
        "by [`render_prompts.py`](render_prompts.py), so it cannot drift out of sync "
        "with the actual prompt source -- if you edit either source file, re-run "
        "`python prompts/render_prompts.py` to regenerate this doc."
    )
    lines.append("")
    lines.append(
        "**These are provided as a guide and source of inspiration for prompt "
        "design, not as a generic, plug-and-play template.** If you are adapting "
        "this framework for a different classification task, you will need to "
        "write your own prompts tailored to your own taxonomy and research "
        "questions."
    )
    lines.append("")
    lines.append(
        "Placeholders like `<INSTRUMENT_SUMMARY>` mark where per-document content "
        "(document summaries, retrieved chunks, few-shot examples) is substituted "
        "in at runtime -- everything else is the fixed instruction text/schema "
        "actually sent to the model."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    lines.append("- [System prompts](#system-prompts)")
    lines.append("- [Summarization prompts](#summarization-prompts)")
    lines.append("- [Classification prompts (questions 2-6)](#classification-prompts-questions-2-6)")
    lines.append("- [Evaluation / judge prompts](#evaluation--judge-prompts)")
    lines.append("")

    # --- System prompts ---
    lines.append("## System prompts")
    lines.append("")
    lines.append("### Summarization system prompt")
    lines.append("")
    lines.append(code_block(p.SYSTEM_PROMPT))
    lines.append("### Classification system prompt")
    lines.append("")
    lines.append(code_block(p.CLASSIFICATION_SYSTEM_PROMPT))

    # --- Summarization prompts (one per focus area, via get_all_prompts) ---
    lines.append("## Summarization prompts")
    lines.append("")
    lines.append(
        "Generated per document via `get_all_prompts(doc_text)`, one per focus area, "
        "using `summarizing_prompt()`:"
    )
    lines.append("")
    focus_areas = [
        ("Sectoral focus (question 2)", "sectoral focus", p.question_2_json_schema, None),
        ("Subject of intervention (question 3)", "subject of intervention", p.question_3_json_schema, p.question_3_note),
        ("Market failure (question 4)", "market failure", p.question_4_json_schema, None),
        ("Type of instrument (question 5)", "type of instrument", p.question_5_json_schema, None),
        ("Metadata and logistical details (question 6)", "metadata and logistical details", p.question_6_json_schema, p.question_6_note),
    ]
    for title, focus_area, schema, note in focus_areas:
        lines.append(f"### {title}")
        lines.append("")
        prompt = p.summarizing_prompt(DOC_TEXT, focus_area, schema, note)
        lines.append(code_block(prompt))

    # --- Classification prompts ---
    lines.append("## Classification prompts (questions 2-6)")
    lines.append("")

    lines.append("### Question 2 -- sectoral focus")
    lines.append("")
    lines.append(f"Short form: {p.short_question_2_prompt}")
    lines.append("")
    lines.append(code_block(p.get_question_2_prompt(INSTRUMENT_SUMMARY, RELEVANT_CHUNKS, EXAMPLES)))

    lines.append("### Question 3 -- subject of intervention")
    lines.append("")
    lines.append(f"Short form: {p.short_question_3_prompt}")
    lines.append("")
    lines.append(code_block(p.get_question_3_prompt(INSTRUMENT_SUMMARY, RELEVANT_CHUNKS, EXAMPLES)))

    lines.append("### Question 4 -- market failure addressed")
    lines.append("")
    lines.append(f"Short form: {p.short_question_4_prompt}")
    lines.append("")
    lines.append(code_block(p.get_question_4_prompt(INSTRUMENT_SUMMARY, RELEVANT_CHUNKS, EXAMPLES)))

    lines.append("### Question 5 -- type of incentive/instrument established")
    lines.append("")
    lines.append(f"Short form: {p.short_question_5_prompt}")
    lines.append("")
    lines.append(code_block(p.get_question_5_prompt(INSTRUMENT_SUMMARY, RELEVANT_CHUNKS, EXAMPLES)))

    lines.append("### Question 6 -- metadata and logistical details")
    lines.append("")
    lines.append(f"Short form: {p.short_question_6_prompt}")
    lines.append("")
    lines.append(code_block(p.get_question_6_prompt(INSTRUMENT_SUMMARY, RELEVANT_CHUNKS)))

    # --- Judge / evaluation prompts ---
    lines.append("## Evaluation / judge prompts")
    lines.append("")
    lines.append(
        "Used to score classification responses against human-labeled ground "
        "truth. An exact string match (after normalizing case/whitespace) short-"
        "circuits this and scores 1.0 automatically without calling the judge."
    )
    lines.append("")
    lines.append("### Judge system prompt")
    lines.append("")
    lines.append(code_block(j.JUDGE_SYSTEM_PROMPT))
    lines.append("### Judge prompt")
    lines.append("")
    lines.append(
        code_block(
            j.get_judge_prompt(
                "<QUESTION_DESCRIPTION, e.g. 'Categorize whether the instrument "
                "primarily targets the financial sector or the real economy'>",
                "<GROUND_TRUTH_ANSWER>",
                "<MODEL_RESPONSE_BEING_EVALUATED>",
            )
        )
    )
    lines.append(
        "### Correction / second-opinion prompt (optional review step)"
    )
    lines.append("")
    lines.append("System prompt for this step:")
    lines.append("")
    lines.append(code_block(j.CORRECTION_SYSTEM_PROMPT))
    lines.append("User prompt:")
    lines.append("")
    lines.append(
        code_block(
            j.get_correction_prompt(
                "<SYSTEM_PROMPT_USED_FOR_THE_ORIGINAL_CLASSIFICATION_CALL>",
                "<USER_PROMPT_USED_FOR_THE_ORIGINAL_CLASSIFICATION_CALL_TRUNCATED_TO_2000_CHARS>",
                "<MODEL_RESPONSE_BEING_REVIEWED>",
            )
        )
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    output_path = Path(__file__).parent / "PROMPTS.md"
    output_path.write_text(render(), encoding="utf-8")
    print(f"Wrote {output_path}")
