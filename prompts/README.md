# Prompts Folder

This folder contains the prompts (classification, summarization, evaluation/judge) used in this project, along with tools and examples for analyzing legal documents using LLMs and RAG (Retrieval-Augmented Generation). The workflow enables you to extract structured information from documents through summarization and classification.

## Overview

The analysis workflow consists of four main steps:

1. **Embedding & Chunking**: Create vector embeddings for RAG
2. **Summarization**: Generate question-focused summaries for each document
3. **Classification**: Classify documents using RAG (optional, project-specific)
4. **Evaluation**: Score classification predictions against a groundtruth set (optional, bring your own data)

## Workflow

### Step 1: Embedding & Chunking

Use `utils/embed.py` to create vector embeddings from your document text files:

```bash
# For a specific jurisdiction
python utils/embed.py --input data/jurisdiction1 --output data/vector_store/vector_store.pkl

```

**What it does:**
- Loads all `.txt` files from the input directory
- Chunks documents into overlapping segments (default: 200 words, 75 word overlap)
- Creates embeddings using OpenAI's embedding models (default: `text-embedding-3-large`)
- Stores embeddings and metadata in a pickle file for later retrieval

**Options:**
- `--chunk-size`: Number of words per chunk (default: 200)
- `--chunk-overlap`: Number of words to overlap between chunks (default: 75)
- `--embedding-model`: OpenAI embedding model to use
- `--batch-size`: Batch size for embedding generation (default: 64)
- `--trim-content`: Trim boilerplate/navigation content from documents

**Output:** A pickle file containing the vector store with document embeddings and chunks.

### Step 2: Summarization

Use `utils/summarize.py` to generate question-focused summaries:

```bash
# For a specific jurisdiction
python utils/summarize.py \
  --input data/jurisdiction1 \
  --output data/summaries \
  --prompts-module prompts.prompts_example
```

**What it does:**
- Loads text files from the input directory
- For each document, generates multiple summaries focused on different questions/aspects
- Handles token limits by truncating documents if needed
- Saves summaries in subdirectories: `data/summaries/{document_name}/question_{N}_summary.txt`

**Requirements:**
- A prompts module that provides:
  - `SYSTEM_PROMPT`: System prompt for the LLM
  - `get_all_prompts(doc_text)`: Function that returns a list of prompts

**Options:**
- `--prompts-module`: Python module path (e.g., `prompts.prompts_example`)
- `--max-tokens`: Maximum context window size (default: 128000)
- `--safety-margin`: Tokens to reserve for system messages (default: 1024)
- `--model`: Model to use for summarization (default: `gpt-4o-mini`)
- `--skip-existing`: Skip documents that already have summaries

**See `prompts_example.py`** in this folder for an example of how to structure your prompts module.

### Step 3: Classification (Optional)

Classification is project-specific and depends on your use case. The general approach is:

1. Use RAG to retrieve relevant document chunks for each classification question
2. Use LLMs to classify documents based on summaries and retrieved chunks
3. Optionally implement a correction loop (e.g., two-LLM approach) for improved accuracy

**Utilities available:**
- `utils/rag.py`: Functions for loading vector stores and querying documents
  - `load_vector_store(path)`: Load a vector store from a pickle file
  - `query_document(store, client, doc_name, query, top_k)`: Retrieve top-k most similar chunks

### Step 4: Evaluation (Optional)

Use `prompts/evaluate.py` to score classification predictions against a human-labeled groundtruth set — real precision, recall, and F1 per question, with Wilson confidence intervals, rather than relying on an LLM-as-judge pass/fail score alone.

```bash
python prompts/evaluate.py \
  --groundtruth data/my_groundtruth.csv \
  --predictions data/my_predictions.csv \
  --taxonomy prompts/evaluate_taxonomy_example.json \
  --questions question_2,question_3 \
  --output data/evaluation_results
```

Or via the Makefile: `make evaluate EVAL_GROUNDTRUTH=... EVAL_PREDICTIONS=... EVAL_QUESTIONS=question_2,question_3`.

**You must bring your own groundtruth and predictions.** `data/groundtruth_example.xlsx` only demonstrates the expected column shape — its label columns are empty — and no example predictions file is bundled at all. There's no way around manually labeling a sample of documents yourself first.

**This is not a fully objective metric, and the script says so loudly.** Precision/recall against a fixed taxonomy assumes one right answer per document per question. In practice, many classification schemes allow (or require) a document to be tagged along more than one dimension at once, categories can genuinely overlap in scope, and a "wrong" prediction can just as easily be a defensible alternative reading as a real model error. When we used this approach on the climate-finance classification task this repository accompanies (14-17 document groundtruth set), a manual read-through of a sample of the disagreements found that a substantial share — roughly half, in our sample — reflected genuine ambiguity rather than model error. `evaluate.py` writes out every individual disagreement (`per_document.csv`, `parse_trace.csv`) specifically so you can do that same manual check on your own data before trusting the topline numbers. Report confidence intervals, not bare percentages, especially at small sample sizes — the script computes Wilson intervals for you.

**Taxonomy config**: `evaluate.py` needs a JSON file describing your canonical label set per question — see `prompts/evaluate_taxonomy_example.json` (built from the real Q2-Q5 schema used in this study) for the format. Two label shapes are supported per question: `"paired"` (category + value, e.g. `"Entities: Non-financial corporations"`) and flat (bare strings). Optional `aliases` and `shorthand_expansions` let you handle known groundtruth abbreviations without loosening the fuzzy-match threshold globally.

## Example Files

### `prompts_example.py`

These are the exact prompts used in the published study this repository accompanies (climate finance policy classification), not a generic template. It includes:

- System prompts for summarization and classification
- JSON schemas for different question types
- Functions to generate prompts for each question
- A `get_all_prompts(doc_text)` function that returns a list of prompts

**Note:** Provided as a guide/source of inspiration for prompt design. You should create your own prompts module tailored to your use case, using this as a reference.

### `judge_prompts.py`

The corresponding evaluation/judge prompts used to score classification responses against ground truth, plus the optional correction/second-opinion prompt. Same caveat as above: these are the real prompts from the study, provided as a reference rather than a generic template.

### `PROMPTS.md`

A readable, rendered view of every prompt in `prompts_example.py` and `judge_prompts.py`, generated by `render_prompts.py`. Regenerate it (`python prompts/render_prompts.py`) whenever either source file changes, rather than editing it by hand, so it can't drift out of sync with the actual prompts.

**Note:** `groundtruth_example.xlsx` (an example ground truth data structure) now lives in `data/groundtruth_example.xlsx`, not in this folder. Its label columns are empty — see the Evaluation section above.

### `evaluate_taxonomy_example.json`

The taxonomy config for `evaluate.py`, built directly from `prompts_example.py`'s Q2-Q5 schemas so it's a real, working example. Same caveat as the other example files: it's the real study's taxonomy, provided as a reference for the config format, not a generic template — write your own taxonomy config for your own classification questions.

## Dependencies

- `openai` - LLM API calls (via OpenRouter)
- `numpy` - Vector operations
- `pickle` - Vector store serialization
- `tiktoken` - Token counting for context window management
- `pandas` - Data handling (for classification workflows)

## Notes

- The workflow uses OpenRouter API for LLM access (set `OPENROUTER_API_KEY` in your `.env` file)
- Vector store uses OpenAI embeddings (default: `text-embedding-3-large`)
- Summarization uses GPT-4o-mini by default for cost efficiency
- The prompts module pattern allows you to customize the summarization/classification questions for your specific use case

## Integration with Data Pipeline

This analysis workflow works with documents prepared using the data pipeline:

1. Documents are scraped and extracted using `utils/scraping.py` (saves as `.txt` files in jurisdiction-specific folders like `data/jurisdiction1/`)
2. Documents are embedded using `utils/embed.py` (creates vector store)
3. Documents are summarized using `utils/summarize.py` (creates question-focused summaries)
4. Documents can be classified using custom scripts that use `utils/rag.py` for retrieval

**Note:** The data pipeline creates jurisdiction-specific folders (e.g., `data/canada/`, `data/jurisdiction1/`) rather than a single `data/scraped_documents/` folder. You can process each jurisdiction separately or combine them as needed.

See the `data/README.md` for details on the data preparation pipeline.
