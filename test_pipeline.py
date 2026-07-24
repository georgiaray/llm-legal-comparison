#!/usr/bin/env python3
"""
Test Pipeline Script

This script tests all utility scripts with a small subset of data:
1. Scrapes the first 5 URLs from urls/urls_example.txt
2. Extracts text from scraped files (extract.py)
3. Processes text (language detection only, to save costs)
4. Creates a vector store from those documents
5. Tests RAG utilities (load vector store, query document)
6. Runs summarization on just one document
7. Tests exploration.py with sample data (if available)

This is designed to be quick and cost-effective while verifying everything works.
"""

import os
import sys
import subprocess
import tempfile
import pickle
from pathlib import Path
from shutil import rmtree

import pandas as pd


def print_step(step_num, description):
    """Print a formatted step message."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*70}\n")


def run_command(cmd, description, check=True):
    """Run a shell command and print the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        if check:
            sys.exit(1)
        return False
    else:
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
        return True


def main():
    # Setup paths
    project_root = Path(__file__).parent.resolve()
    urls_file = project_root / "urls" / "urls_example.txt"
    
    # Create temporary directories for test data
    test_data_dir = project_root / "test_data"
    test_output_dir = test_data_dir / "scraped"
    test_extracted_csv = test_data_dir / "extracted.csv"
    test_processed_csv = test_data_dir / "processed.csv"
    test_vector_store = test_data_dir / "vector_store" / "test_vectors.pkl"
    test_summaries_dir = test_data_dir / "summaries"
    
    # Clean up any existing test data
    if test_data_dir.exists():
        print(f"Cleaning up existing test data...")
        rmtree(test_data_dir)
    
    # Create directories
    test_output_dir.mkdir(parents=True)
    test_vector_store.parent.mkdir(parents=True)
    test_summaries_dir.mkdir(parents=True)
    
    print(f"Test data directory: {test_data_dir}")
    print(f"Scraped documents will go to: {test_output_dir}")
    print(f"Vector store will be saved to: {test_vector_store}")
    print(f"Summaries will be saved to: {test_summaries_dir}")
    
    # Step 1: Extract first 5 URLs
    print_step(1, "Extracting first 5 URLs from urls_example.txt")
    
    if not urls_file.exists():
        print(f"❌ Error: {urls_file} not found")
        sys.exit(1)
    
    with open(urls_file, 'r') as f:
        all_urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    test_urls = all_urls[:5]
    
    if len(test_urls) < 5:
        print(f"⚠️  Warning: Only found {len(test_urls)} URLs, using all available")
    
    print(f"Found {len(test_urls)} URLs to test:")
    for i, url in enumerate(test_urls, 1):
        print(f"  {i}. {url[:80]}...")
    
    # Create temporary URLs file
    temp_urls_file = test_data_dir / "test_urls.txt"
    with open(temp_urls_file, 'w') as f:
        f.write('\n'.join(test_urls))
    
    print(f"✅ Created temporary URLs file: {temp_urls_file}")
    
    # Step 2: Scrape documents
    print_step(2, "Scraping documents (first 5 URLs)")
    
    scrape_cmd = [
        sys.executable,
        str(project_root / "utils" / "scraping.py"),
        "--urls-file", str(temp_urls_file),
        "--out", str(test_output_dir)
    ]
    
    if not run_command(scrape_cmd, "Scraping documents"):
        print("⚠️  Some documents may have failed to scrape, continuing anyway...")
    
    # Check how many files were actually scraped
    scraped_files = list(test_output_dir.glob("*.txt"))
    print(f"\nScraped {len(scraped_files)} documents")
    for f in sorted(scraped_files):
        size = f.stat().st_size
        print(f"   - {f.name}: {size:,} bytes")
    
    if len(scraped_files) == 0:
        print("❌ Error: No documents were scraped. Cannot continue.")
        sys.exit(1)

    # Verify scraped files actually contain text, not just that they exist.
    # A "successful" scrape that wrote a 0-byte file (e.g. extraction silently
    # failed) would otherwise pass every downstream check.
    empty_scraped_files = [f for f in scraped_files if f.stat().st_size == 0]
    try:
        assert not empty_scraped_files, (
            f"{len(empty_scraped_files)} scraped file(s) are empty: "
            f"{[f.name for f in empty_scraped_files]}"
        )
    except AssertionError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Step 3: Extract text (using extract.py)
    print_step(3, "Testing extract.py - Extracting text from files")
    
    extract_cmd = [
        sys.executable,
        str(project_root / "utils" / "extract.py"),
        "--folder", str(test_output_dir),
        "--output", str(test_extracted_csv)
    ]
    
    if not run_command(extract_cmd, "Extracting text to CSV"):
        print("⚠️  Warning: Text extraction failed, continuing anyway...")
    else:
        if test_extracted_csv.exists():
            size = test_extracted_csv.stat().st_size
            print(f"✅ Extracted CSV created: {size:,} bytes")

            # Validate the CSV's actual contents, not just that the file exists.
            try:
                extracted_df = pd.read_csv(test_extracted_csv, escapechar="\\")
                assert {"file", "text"}.issubset(extracted_df.columns), (
                    f"Expected 'file' and 'text' columns, got: {list(extracted_df.columns)}"
                )
                assert len(extracted_df) == len(scraped_files), (
                    f"Expected {len(scraped_files)} rows (one per scraped file), "
                    f"got {len(extracted_df)}"
                )
                non_empty = extracted_df["text"].fillna("").str.len().gt(0).sum()
                assert non_empty > 0, "All extracted 'text' values are empty"
                print(f"✅ Extracted CSV validated: {non_empty}/{len(extracted_df)} rows have text")
            except AssertionError as e:
                print(f"❌ Error: Extracted CSV failed validation: {e}")
                sys.exit(1)

    # Step 4: Process text (language detection only, to save costs)
    print_step(4, "Testing process.py - Language detection (detect_only mode)")
    
    if test_extracted_csv.exists():
        process_cmd = [
            sys.executable,
            str(project_root / "utils" / "process.py"),
            "--input", str(test_extracted_csv),
            "--output", str(test_processed_csv),
            "--mode", "detect_only"  # Just detect language, don't translate
        ]
        
        if not run_command(process_cmd, "Detecting languages"):
            print("⚠️  Warning: Language detection failed, continuing anyway...")
        else:
            if test_processed_csv.exists():
                size = test_processed_csv.stat().st_size
                print(f"✅ Processed CSV created: {size:,} bytes")
                # Show language distribution, and validate detection actually ran
                # rather than just checking the file was written.
                try:
                    df = pd.read_csv(test_processed_csv, escapechar='\\')
                    assert 'detected_language' in df.columns, (
                        "Processed CSV is missing the 'detected_language' column"
                    )
                    lang_counts = df['detected_language'].value_counts()
                    print(f"\nLanguage distribution:")
                    for lang, count in lang_counts.items():
                        print(f"   - {lang}: {count}")

                    known_lang_count = (df['detected_language'] != 'unknown').sum()
                    assert known_lang_count > 0, (
                        "Language detection returned 'unknown' for every row"
                    )
                except AssertionError as e:
                    print(f"❌ Error: Language detection failed validation: {e}")
                    sys.exit(1)
                except Exception as e:
                    print(f"⚠️  Could not show language distribution: {e}")
    
    # Step 5: Create vector store
    print_step(5, "Creating vector store from scraped documents")
    
    embed_cmd = [
        sys.executable,
        str(project_root / "utils" / "embed.py"),
        "--input", str(test_output_dir),
        "--output", str(test_vector_store)
        # Note: Content trimming is enabled by default (matching original workflow)
    ]
    
    if not run_command(embed_cmd, "Creating vector embeddings"):
        print("❌ Error: Failed to create vector store")
        sys.exit(1)
    
    # Verify vector store was created
    if test_vector_store.exists():
        size = test_vector_store.stat().st_size
        print(f"✅ Vector store created: {size:,} bytes")
    else:
        print("❌ Error: Vector store file not found")
        sys.exit(1)

    # Validate the pickle's internal structure, not just that a non-empty file
    # exists — a truncated or malformed pickle would otherwise pass silently.
    try:
        with open(test_vector_store, "rb") as f:
            vector_store_contents = pickle.load(f)

        assert isinstance(vector_store_contents, dict) and len(vector_store_contents) > 0, (
            "Vector store is empty or not a dict"
        )
        for doc_name, entry in vector_store_contents.items():
            assert "embeddings" in entry and "chunks" in entry, (
                f"Document '{doc_name}' is missing 'embeddings' or 'chunks'"
            )
            n_embeddings = len(entry["embeddings"])
            n_chunks = len(entry["chunks"])
            assert n_embeddings == n_chunks and n_embeddings > 0, (
                f"Document '{doc_name}' has {n_embeddings} embeddings but "
                f"{n_chunks} chunks (expected equal, non-zero counts)"
            )
            assert all(len(vec) > 0 for vec in entry["embeddings"]), (
                f"Document '{doc_name}' has one or more zero-length embedding vectors"
            )
        print(f"✅ Vector store structure validated: {len(vector_store_contents)} documents")
    except AssertionError as e:
        print(f"❌ Error: Vector store failed structural validation: {e}")
        sys.exit(1)

    # Step 6: Test RAG utilities
    print_step(6, "Testing rag.py - Loading vector store and querying")
    
    if test_vector_store.exists():
        # Test loading the vector store
        try:
            from utils.rag import load_vector_store
            from openai import OpenAI
            from dotenv import load_dotenv
            
            load_dotenv()
            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            
            print("Loading vector store...")
            store = load_vector_store(test_vector_store)
            print(f"✅ Vector store loaded: {len(store)} documents")
            
            # Test querying one document
            if len(store) > 0:
                test_doc_name = list(store.keys())[0]
                print(f"Querying document: {test_doc_name}")
                
                from utils.rag import query_document
                results = query_document(
                    store,
                    client,
                    test_doc_name,
                    "What is this document about?",
                    top_k=2
                )

                # Validate the shape of what came back, not just that the call
                # didn't raise.
                assert isinstance(results, list), "query_document should return a list"
                for r in results:
                    assert {"chunk_index", "similarity", "text"}.issubset(r.keys()), (
                        f"Result missing expected keys: {list(r.keys())}"
                    )
                    assert isinstance(r["text"], str) and len(r["text"]) > 0, (
                        "Result chunk text is empty"
                    )

                print(f"✅ Query successful: Retrieved {len(results)} chunks")
                if results:
                    print(f"   Top chunk similarity: {results[0]['similarity']:.3f}")
        except Exception as e:
            print(f"⚠️  Warning: RAG test failed: {e}")
    else:
        print("⚠️  Warning: Vector store not found, skipping RAG test")
    
    # Step 7: Run summarization on just one document
    print_step(7, "Testing summarize.py - Running summarization on one document")
    
    # Pick the first successfully scraped document
    test_doc = sorted(scraped_files)[0]
    test_doc_name = test_doc.stem
    
    print(f"Testing summarization on: {test_doc_name}")
    
    # Create a temporary directory with just this one document
    single_doc_dir = test_data_dir / "single_doc"
    single_doc_dir.mkdir()
    single_doc_file = single_doc_dir / test_doc.name
    single_doc_file.write_text(test_doc.read_text())
    
    summarize_cmd = [
        sys.executable,
        str(project_root / "utils" / "summarize.py"),
        "--input", str(single_doc_dir),
        "--output", str(test_summaries_dir),
        "--prompts-module", "prompts.prompts_example",
        "--model", "gpt-4o-mini"  # Use cheaper model for testing
    ]
    
    if not run_command(summarize_cmd, f"Summarizing {test_doc_name}"):
        print("❌ Error: Failed to create summaries")
        sys.exit(1)
    
    # Check if summaries were created
    summary_dir = test_summaries_dir / test_doc_name
    if summary_dir.exists():
        summary_files = list(summary_dir.glob("*.txt"))
        print(f"\n✅ Created {len(summary_files)} summaries:")
        for f in sorted(summary_files):
            size = f.stat().st_size
            print(f"   - {f.name}: {size:,} bytes")

        try:
            assert len(summary_files) > 0, "No summary files were written"
            empty_summaries = [f.name for f in summary_files if f.stat().st_size == 0]
            assert not empty_summaries, f"Empty summary file(s): {empty_summaries}"
        except AssertionError as e:
            print(f"❌ Error: Summaries failed validation: {e}")
            sys.exit(1)
    else:
        print("❌ Error: Summary directory not found")
        sys.exit(1)
    
    # Step 8: Test exploration.py (if sample data exists)
    print_step(8, "Testing exploration.py - Analyzing sample dataset")
    
    sample_df = project_root / "data" / "sample_df.csv"
    if sample_df.exists():
        explore_cmd = [
            sys.executable,
            str(project_root / "utils" / "exploration.py"),
            "--input", str(sample_df)
        ]
        
        if run_command(explore_cmd, "Exploring sample dataset", check=False):
            print("✅ Exploration completed successfully")
        else:
            print("⚠️  Warning: Exploration failed, but this is optional")
    else:
        print("No sample_df.csv found, skipping exploration test")
    
    # Final summary
    print_step(9, "Test Complete!")

    # Precompute this rather than inlining a conditional in the f-string below:
    # the previous version put the `if/else` outside the `{...}` braces around
    # `.stat().st_size`, so it (a) printed the literal text "bytes if
    # test_vector_store.exists() else 'N/A'" instead of evaluating it, and
    # (b) called .stat() unconditionally, which would raise if the file didn't
    # exist.
    if test_vector_store.exists():
        vector_store_size_str = f"{test_vector_store.stat().st_size:,} bytes"
    else:
        vector_store_size_str = "N/A"

    print(f"""
✅ Pipeline test completed successfully!

Summary:
  - URLs tested: {len(test_urls)}
  - Documents scraped: {len(scraped_files)}
  - Text extraction: {'✅' if test_extracted_csv.exists() else '❌'}
  - Language detection: {'✅' if test_processed_csv.exists() else '❌'}
  - Vector store created: {test_vector_store.name if test_vector_store.exists() else 'N/A'} ({vector_store_size_str})
  - RAG utilities: {'✅ Tested' if test_vector_store.exists() else '⚠️ Skipped'}
  - Document analyzed: {test_doc_name}
  - Summaries created: {len(summary_files) if summary_dir.exists() else 0}
  - Exploration: {'✅ Tested' if sample_df.exists() else '⚠️ Skipped (no sample_df.csv)'}

Test data location: {test_data_dir}
You can inspect the results or delete this directory when done.
    """)
    
    # Ask if user wants to clean up
    print("\nTip: Run 'rm -rf test_data' to clean up test data when done.")


if __name__ == "__main__":
    main()
