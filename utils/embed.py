#!/usr/bin/env python3
"""
Document Embedding and Chunking Script

This script chunks documents and creates vector embeddings for RAG (Retrieval-Augmented Generation).
It processes text files, splits them into overlapping chunks, generates embeddings, and saves
them to a vector store for later retrieval.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Default location of the boilerplate pattern config used by trim_non_content().
DEFAULT_BOILERPLATE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "boilerplate_patterns.json"

# Minimal built-in fallback, used only if the config file above can't be found/
# loaded, so trim_non_content() still degrades gracefully rather than failing.
_FALLBACK_PATTERNS = {
    "default": {
        "head_patterns": [r"^Skip to main content[\n\r]+", r"^Skip to [^\n]+\n"],
        "tail_patterns": [r"\nDate modified:[^\n]*$"],
        "keywords": ["Back to top", "Contact us"],
    }
}


def load_boilerplate_patterns(jurisdiction: str = "default", config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load head/tail regex patterns and non-content keywords for a given
    jurisdiction from the boilerplate patterns config file.

    Args:
        jurisdiction: Key into the config file (e.g. "default", "canada").
            Add your own jurisdiction entry to the config file rather than
            hardcoding new patterns in this module -- see config/README.md.
        config_path: Path to the JSON config file. Defaults to
            config/boilerplate_patterns.json at the project root.

    Returns:
        Dict with 'head_patterns', 'tail_patterns', and 'keywords' lists.
    """
    path = config_path or DEFAULT_BOILERPLATE_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_patterns = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load boilerplate config from {path} ({e}); using minimal built-in fallback patterns.")
        all_patterns = _FALLBACK_PATTERNS
        jurisdiction = "default"

    if jurisdiction not in all_patterns:
        available = [k for k in all_patterns.keys() if not k.startswith("_")]
        raise KeyError(f"Unknown jurisdiction '{jurisdiction}'. Available: {available}. Add it to {path} to define new patterns.")

    entry = all_patterns[jurisdiction]
    return {
        "head_patterns": entry.get("head_patterns", []),
        "tail_patterns": entry.get("tail_patterns", []),
        "keywords": entry.get("keywords", []),
    }


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 75) -> List[str]:
    """
    Split a document into overlapping word chunks of fixed size.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        chunk_overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    total_words = len(words)

    while start < total_words:
        end = min(total_words, start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= total_words:
            break
        # Decide new start: handle overlap
        start += max(1, chunk_size - chunk_overlap)

    return chunks


def trim_non_content(text: str, jurisdiction: str = "default", config_path: Optional[Path] = None) -> str:
    """
    Attempts to trim repeated navigation/boilerplate 'chrome' text
    like headers, navbars, footers, menus, and non-content at the head and tail of the doc.
    Not guaranteed to catch everything, but tries to remove common patterns.

    Patterns are loaded from a config file (config/boilerplate_patterns.json by
    default) rather than hardcoded, since boilerplate text is inherently
    jurisdiction/site-specific. The "default" jurisdiction is a conservative,
    generic set; a "canada" jurisdiction is also provided (the exact patterns
    this function used to hardcode for Canada.ca pages). Add your own
    jurisdiction to the config file for other sites -- see config/README.md.

    Args:
        text: Text to clean
        jurisdiction: Which pattern set to use (default: "default", generic).
            Pass "canada" for the Canada.ca-specific patterns, or add your own
            entry to the config file for other jurisdictions/sites.
        config_path: Optional path to a boilerplate patterns JSON config file,
            overriding the default location.

    Returns:
        Cleaned text
    """
    patterns = load_boilerplate_patterns(jurisdiction, config_path)
    head_patterns = patterns["head_patterns"]
    tail_patterns = patterns["tail_patterns"]
    non_content_keywords = patterns["keywords"]

    cleaned = text.strip()

    # Heuristically trim head
    for pat in head_patterns:
        cleaned_new = re.sub(pat, '', cleaned, flags=re.IGNORECASE|re.MULTILINE)
        if len(cleaned_new) < len(cleaned) - 8:  # Actually shortened content?
            cleaned = cleaned_new
            break

    # Heuristically trim tail
    trimmed = False
    for pat in tail_patterns:
        cleaned_new = re.sub(pat, '', cleaned, flags=re.IGNORECASE|re.MULTILINE)
        if len(cleaned_new) < len(cleaned) - 8:
            cleaned = cleaned_new
            trimmed = True
    # Try tail removal twice (to catch two-stage footers)
    if trimmed:
        for pat in tail_patterns:
            cleaned_new = re.sub(pat, '', cleaned, flags=re.IGNORECASE|re.MULTILINE)
            if len(cleaned_new) < len(cleaned) - 8:
                cleaned = cleaned_new

    # Secondary: For repeated menu/footer junk, try to cut on keyword.
    # Remove lines at top or bottom containing only these keywords
    lines = cleaned.splitlines()
    # Remove leading non-content lines
    while lines and any(k.lower() in lines[0].lower() for k in non_content_keywords):
        lines = lines[1:]
    # Remove trailing non-content lines
    while lines and any(k.lower() in lines[-1].lower() for k in non_content_keywords):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def load_documents(data_dir: Path) -> Dict[str, str]:
    """
    Read every .txt file in the input directory.
    
    Args:
        data_dir: Directory containing text files
    
    Returns:
        Dictionary mapping document names (without extension) to their text content
    """
    documents: Dict[str, str] = {}
    for path in sorted(data_dir.glob("*.txt")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def embed_chunk_batch(client: OpenAI, batch: List[Dict[str, Any]], embedding_model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks.
    
    Args:
        client: OpenAI client instance
        batch: List of chunk dictionaries with 'text' key
        embedding_model: Model to use for embeddings
    
    Returns:
        List of embedding vectors
    """
    inputs = [item["text"] for item in batch]
    response = client.embeddings.create(
        model=embedding_model,
        input=inputs
    )
    # OpenAI returns embeddings in the same order as inputs
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(
        description="Create vector embeddings for documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings from documents in a folder
  python utils/embed.py --input data/scraped_documents --output data/vector_store/vector_store.pkl
  
  # Custom chunk size and overlap
  python utils/embed.py --input data/documents --output data/vectors.pkl --chunk-size 300 --chunk-overlap 100

  # Use the Canada.ca-specific boilerplate patterns instead of the generic default
  python utils/embed.py --input data/canada --output data/vectors.pkl --jurisdiction canada
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing text files to embed"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for vector store pickle file"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=75,
        help="Number of words to overlap between chunks (default: 75)"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model to use (default: text-embedding-3-large)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)"
    )
    parser.add_argument(
        "--no-trim-content",
        action="store_true",
        help="Skip trimming boilerplate/navigation content from documents (trimming is enabled by default)"
    )
    parser.add_argument(
        "--jurisdiction",
        default="default",
        help="Boilerplate pattern set to use for --trim-content, from config/boilerplate_patterns.json "
             "(default: 'default', a generic/conservative set; 'canada' is also provided for Canada.ca pages)"
    )
    parser.add_argument(
        "--boilerplate-config",
        default=None,
        help="Path to a custom boilerplate patterns JSON config file (default: config/boilerplate_patterns.json)"
    )

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("Please set OPENROUTER_API_KEY in your .env file or environment variables")
    
    # Setup paths
    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Load documents
    print(f"Loading documents from: {input_dir}")
    documents = load_documents(input_dir)
    print(f"Loaded {len(documents)} documents")
    
    # Trim content (enabled by default; jurisdiction-configurable, see config/boilerplate_patterns.json)
    if not args.no_trim_content:
        print(f"Trimming boilerplate content (jurisdiction: {args.jurisdiction})...")
        boilerplate_config_path = Path(args.boilerplate_config) if args.boilerplate_config else None
        for doc_name in documents:
            documents[doc_name] = trim_non_content(
                documents[doc_name],
                jurisdiction=args.jurisdiction,
                config_path=boilerplate_config_path,
            )
    
    # Chunk documents
    print(f"Chunking documents (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunk_records: List[Dict[str, Any]] = []
    for doc_name, text in documents.items():
        chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "doc_name": doc_name,
                    "chunk_index": idx,
                    "text": chunk,
                }
            )
    
    print(f"Prepared {len(chunk_records)} chunks across all documents")
    
    if not chunk_records:
        print("No chunks were prepared; vector store was not created.")
        return
    
    # Generate embeddings
    print(f"Generating embeddings (batch size: {args.batch_size})...")
    enriched_records: List[Dict[str, Any]] = []
    for start in range(0, len(chunk_records), args.batch_size):
        batch = chunk_records[start:start + args.batch_size]
        embeddings = embed_chunk_batch(client, batch, args.embedding_model)
        for record, embedding in zip(batch, embeddings):
            enriched = {**record, "embedding": embedding}
            enriched_records.append(enriched)
        print(f"Embedded {len(enriched_records)} / {len(chunk_records)} chunks", end="\r")
    
    print()  # New line after progress
    
    # Organize per-document and persist
    print("Organizing vector store...")
    vector_store: Dict[str, Dict[str, Any]] = {}
    for record in enriched_records:
        doc_entry = vector_store.setdefault(
            record["doc_name"],
            {"embeddings": [], "chunks": []}
        )
        doc_entry["embeddings"].append(record["embedding"])
        doc_entry["chunks"].append(
            {
                "chunk_index": record["chunk_index"],
                "text": record["text"],
            }
        )
    
    # Convert embeddings to numpy arrays
    for doc_name, doc_entry in vector_store.items():
        doc_entry["embeddings"] = np.array(doc_entry["embeddings"], dtype=np.float32)
    
    # Save vector store
    with open(output_path, "wb") as f:
        pickle.dump(vector_store, f)
    
    print(f"✅ Persisted vector store to {output_path}")
    print(f"   - Documents: {len(vector_store)}")
    print(f"   - Total chunks: {len(enriched_records)}")


if __name__ == "__main__":
    main()
