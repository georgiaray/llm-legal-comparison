# Config Folder

Configuration data for the pipeline that isn't code — currently just the
boilerplate/navigation patterns used by `utils/embed.py`'s `trim_non_content()`
to strip web page "chrome" (menus, footers, skip-links) before chunking and
embedding documents.

## `boilerplate_patterns.json`

Keyed by jurisdiction. Each entry has three lists:

- `head_patterns`: regexes tried against the start of the document; the first
  one that visibly shortens the text is applied.
- `tail_patterns`: regexes tried against the end of the document (applied in
  up to two passes, to catch two-stage footers).
- `keywords`: substrings used to strip leading/trailing lines that consist
  mostly of navigation/footer junk, after the regex passes above.

Two jurisdictions are provided out of the box:

- `default`: a conservative, jurisdiction-agnostic set (things like "Skip to
  main content", generic footer/date-modified lines) that's unlikely to strip
  real content regardless of source. Used automatically unless you specify
  otherwise.
- `canada`: the exact patterns this function used to have hardcoded, for
  Canada.ca / Government of Canada pages (e.g. "Gouvernement du Canada",
  "Report a problem or mistake on this page"). Opt into these with
  `--jurisdiction canada` on `utils/embed.py`, or `trim_non_content(text,
  jurisdiction="canada")` if calling it directly.

**Adding your own jurisdiction:** add a new top-level key to this file (e.g.
`"eu"`, `"brasil"`) with the same three lists, tailored to the boilerplate
patterns of the sites you're scraping from. Don't edit `default` unless the
pattern is genuinely universal — that's what keeps it safe to apply to
documents from any jurisdiction by default.

You can also point at an entirely separate config file instead of editing
this one, via `--boilerplate-config path/to/your_config.json`.
