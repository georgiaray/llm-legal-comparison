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
	@echo "  make evaluate    Not yet implemented -- see FIXES.md item 6"
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
	@echo "evaluate.py has not been implemented yet -- see FIXES.md item 6."
	@echo "Once it exists, wire it up here rather than leaving 'make all' incomplete."

test:
	$(PYTHON) -m pytest

clean:
	@echo "Removing generated outputs for DATASET=$(DATASET) (raw scraped text in $(SCRAPED_DIR) is left alone)..."
	rm -f $(EXTRACTED_CSV) $(PROCESSED_CSV) $(VECTOR_STORE)
	rm -rf $(SUMMARIES_DIR)
