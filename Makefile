# Makefile: one-command entry points for the VectorLaw pipeline.
#
# `make all` runs scrape -> extract -> process -> embed -> summarize against
# the bundled example URLs, so a reviewer can reproduce a full run without
# stitching together five CLI calls by hand. Override variables to point at
# your own data, e.g.:
#
#   make all DATASET=canada URLS_FILE=urls/canada_urls.txt
#
# NOTE on data flow, so this Makefile doesn't misrepresent what the pipeline
# actually does: `embed` and `summarize` both read raw text directly from
# SCRAPED_DIR, independently of `extract`/`process`. The `extract`/`process`
# steps produce a separate CSV (language-detected/translated text) that nothing
# downstream currently consumes automatically -- they're included here as
# real, runnable pipeline stages, not because embed/summarize depend on their
# output. See data/README.md for the full data-preparation write-up.

# --- Overridable configuration ---
DATASET               ?= example
URLS_FILE             ?= urls/urls_example.txt
SCRAPED_DIR            ?= data/$(DATASET)
EXTRACTED_CSV          ?= data/processed/$(DATASET)_extracted.csv
PROCESSED_CSV          ?= data/processed/$(DATASET)_processed.csv
PROCESS_MODE           ?= auto
VECTOR_STORE           ?= data/vector_store/$(DATASET).pkl
SUMMARIES_DIR          ?= data/summaries
CHUNK_SIZE             ?= 200
CHUNK_OVERLAP          ?= 75
BOILERPLATE_JURISDICTION ?= default
PROMPTS_MODULE         ?= prompts.prompts_example
MODEL                  ?= gpt-4o-mini

# `make evaluate` config. There is no usable default for EVAL_GROUNDTRUTH or
# EVAL_PREDICTIONS -- data/groundtruth_example.xlsx only shows the expected
# *shape* of a groundtruth file (its label columns are empty), and no example
# predictions file is bundled at all. You must supply both yourself, e.g.:
#   make evaluate EVAL_GROUNDTRUTH=data/my_groundtruth.csv EVAL_PREDICTIONS=data/my_predictions.csv EVAL_QUESTIONS=question_2,question_3
# See prompts/evaluate.py's module docstring before running this -- in
# particular, precision/recall against a classification taxonomy is not a
# fully objective metric; read that docstring's warning before trusting the
# numbers it prints.
EVAL_GROUNDTRUTH       ?=
EVAL_PREDICTIONS       ?=
EVAL_TAXONOMY          ?= prompts/evaluate_taxonomy_example.json
EVAL_QUESTIONS         ?=
EVAL_JOIN_KEY          ?= unique_id
EVAL_THRESHOLD         ?= 85
EVAL_OUTPUT            ?= data/evaluation_results

PYTHON ?= python3

.PHONY: help all scrape extract process embed summarize evaluate test clean

help:
	@echo "VectorLaw pipeline -- one-command entry points"
	@echo ""
	@echo "  make all         Run scrape -> extract -> process -> embed -> summarize"
	@echo "  make scrape      Download + extract text from URLS_FILE into SCRAPED_DIR"
	@echo "  make extract     Re-extract SCRAPED_DIR into a CSV (file, text columns)"
	@echo "  make process     Language-detect/translate the extracted CSV"
	@echo "  make embed       Chunk + embed SCRAPED_DIR into VECTOR_STORE"
	@echo "  make summarize   Generate question-focused summaries into SUMMARIES_DIR"
	@echo "  make evaluate    Score EVAL_PREDICTIONS against EVAL_GROUNDTRUTH (both required -- no bundled example data)"
	@echo "  make test        Run the pytest unit test suite"
	@echo "  make clean       Remove generated outputs for DATASET (not raw scraped text)"
	@echo ""
	@echo "Override any variable on the command line, e.g.:"
	@echo "  make all DATASET=canada URLS_FILE=urls/canada_urls.txt"
	@echo ""
	@echo "Current settings:"
	@echo "  DATASET=$(DATASET)  URLS_FILE=$(URLS_FILE)  SCRAPED_DIR=$(SCRAPED_DIR)"
	@echo "  PROMPTS_MODULE=$(PROMPTS_MODULE)  MODEL=$(MODEL)"

all: scrape extract process embed summarize

scrape:
	$(PYTHON) utils/scraping.py --urls-file $(URLS_FILE) --out $(SCRAPED_DIR)

extract:
	$(PYTHON) utils/extract.py --folder $(SCRAPED_DIR) --output $(EXTRACTED_CSV)

process: extract
	$(PYTHON) utils/process.py --input $(EXTRACTED_CSV) --output $(PROCESSED_CSV) --mode $(PROCESS_MODE)

embed:
	$(PYTHON) utils/embed.py \
		--input $(SCRAPED_DIR) \
		--output $(VECTOR_STORE) \
		--chunk-size $(CHUNK_SIZE) \
		--chunk-overlap $(CHUNK_OVERLAP) \
		--jurisdiction $(BOILERPLATE_JURISDICTION)

summarize:
	$(PYTHON) utils/summarize.py \
		--input $(SCRAPED_DIR) \
		--output $(SUMMARIES_DIR) \
		--prompts-module $(PROMPTS_MODULE) \
		--model $(MODEL)

evaluate:
ifeq ($(strip $(EVAL_GROUNDTRUTH)),)
	@echo "ERROR: EVAL_GROUNDTRUTH is required and has no default -- you need your own"
	@echo "human-labeled groundtruth file (data/groundtruth_example.xlsx only shows the"
	@echo "expected shape; its label columns are empty). Example:"
	@echo "  make evaluate EVAL_GROUNDTRUTH=data/my_groundtruth.csv EVAL_PREDICTIONS=data/my_predictions.csv EVAL_QUESTIONS=question_2,question_3"
	@exit 1
endif
ifeq ($(strip $(EVAL_PREDICTIONS)),)
	@echo "ERROR: EVAL_PREDICTIONS is required and has no default -- no example"
	@echo "predictions file is bundled with this repo. Point it at your own model"
	@echo "output, in the same shape as EVAL_GROUNDTRUTH (see prompts/evaluate.py --help)."
	@exit 1
endif
ifeq ($(strip $(EVAL_QUESTIONS)),)
	@echo "ERROR: EVAL_QUESTIONS is required, e.g. EVAL_QUESTIONS=question_2,question_3"
	@echo "(must match column names in both EVAL_GROUNDTRUTH and EVAL_PREDICTIONS, and"
	@echo "must exist as keys in EVAL_TAXONOMY=$(EVAL_TAXONOMY))."
	@exit 1
endif
	$(PYTHON) prompts/evaluate.py \
		--groundtruth $(EVAL_GROUNDTRUTH) \
		--predictions $(EVAL_PREDICTIONS) \
		--taxonomy $(EVAL_TAXONOMY) \
		--questions $(EVAL_QUESTIONS) \
		--join-key $(EVAL_JOIN_KEY) \
		--threshold $(EVAL_THRESHOLD) \
		--output $(EVAL_OUTPUT)

test:
	$(PYTHON) -m pytest

clean:
	@echo "Removing generated outputs for DATASET=$(DATASET) (raw scraped text in $(SCRAPED_DIR) is left alone)..."
	rm -f $(EXTRACTED_CSV) $(PROCESSED_CSV) $(VECTOR_STORE)
	rm -rf $(SUMMARIES_DIR)
