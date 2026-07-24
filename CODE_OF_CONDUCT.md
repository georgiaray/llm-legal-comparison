# Code of Conduct

VectorLaw isn't a community project looking for contributors, so this isn't a
typical interpersonal code of conduct. It's a short set of expectations for
anyone using or adapting this software: use it freely, but scrape and handle
data responsibly, and cite the work if it's useful to you.

## Use it freely

VectorLaw is released under the MIT License (see `LICENSE`). You're free to
use, modify, adapt, and redistribute it for any purpose, commercial or
otherwise, subject only to that license. There's no expectation that you
contribute changes back, ask permission, or coordinate with the maintainer.

## Scrape responsibly

`utils/scraping.py` will download whatever URLs you point it at, as fast as
its retry logic allows. It does **not** enforce rate limiting or check
`robots.txt` on your behalf — that's on you. When pointing this tool at any
website, especially government and public-institution sites that other
researchers and the public also rely on:

- Respect `robots.txt` and the site's terms of service.
- Rate-limit your own requests. Don't hammer a server just because nothing
  in this codebase stops you.
- Prefer official bulk-data or API endpoints over scraping rendered pages,
  where they exist (e.g. Climate Policy Radar's dataset exports).
- If a site or database explicitly asks you not to scrape it, don't.

## Data provenance and copyright

This tool extracts and stores the text of whatever documents you feed it.
Before you scrape, store, or redistribute that content:

- **Copyright status varies by jurisdiction and source.** Many government
  legal/policy documents are public domain or covered by an open-government
  licence, but this is not universal — some jurisdictions and some document
  types (e.g. certain regulatory guidance, third-party commentary) retain
  copyright. Check the licence/terms of the specific source before
  republishing or redistributing extracted text, rather than assuming public
  documents are automatically free to reuse.
- **Curated third-party datasets have their own terms.** If you're using a
  dataset like Climate Policy Radar's as a starting point (see
  `data/README.md`), that dataset's own terms of use apply independently of
  this tool's license.
- **Keep a record of where data came from.** The `urls/` and `data/`
  structure this project uses is designed to make it easy to trace processed
  text back to its source URL — keep that trail intact if you plan to publish
  results, so your provenance is auditable.

## Data protection

- Legal and policy documents are usually public by nature, but can
  occasionally include personal information (named individuals, contact
  details of officials, etc.). Be mindful of applicable data protection law
  (e.g. GDPR, PIPEDA) if you reuse, republish, or build datasets from
  extracted text that includes personal data.
- This pipeline sends document text to third-party LLM APIs (via
  OpenRouter/OpenAI) for summarization, embedding, and classification. Don't
  feed it confidential, sensitive, or export-controlled material without
  first checking the relevant provider's data handling and retention terms.
- Keep your `.env` file and API keys out of version control (already
  gitignored) and out of anything you publish.

## Citation

If you use this software in academic or other published work, please cite
it — see `CITATION.cff` for the current citation. Citing the work is the
main thing we ask for in return for the MIT license.
