# Deductive AI Coder — MVP

A transparent Streamlit research prototype for **deductive qualitative coding with a human-in-the-loop calibration workflow**.

The application has three stages:

1. **Protocol Co-Design** — define a study, build/edit a structured deductive codebook manually or with Gemini, optionally grounding co-design in uploaded literature/source files, then freeze a protocol version.
2. **Calibration** — compare Gemini coding with one or two human coders, inspect individual AI decisions on demand, and use disagreement cases to propose protocol refinements.
3. **Full Dataset Coding** — apply a frozen protocol version to a CSV, Excel, or SPSS dataset and export coded data plus a machine-readable audit record.

This repository is intentionally an **MVP rather than a production qualitative-analysis platform**. The code is separated into small modules so the prototype can be tested now and extended later.

## Design principles

- **No study topic is hard-coded.** The codebook is project data, not Python constants.
- **Deductive coding first.** The MVP does not attempt general inductive/thematic analysis.
- **Researchers remain decision-makers.** Gemini proposes/refines; the researcher edits and freezes protocols.
- **Versioned protocols.** Calibration and batch coding use an immutable snapshot. A revised draft must be frozen as a new version.
- **Structured AI output.** Coding decisions are parsed as validated JSON code IDs rather than extracted from free-form prose.
- **Server-side API key.** Test users do not enter or receive the Gemini key.
- **Minimal raw-data retention by the app.** Raw calibration and batch data are kept in the active Streamlit session and are not embedded in downloadable project JSON files.

## Repository structure

```text
.
├── app.py                          # Streamlit interface / workflow
├── requirements.txt
├── DESIGN_NOTES.md                 # MVP scope and architecture rationale
├── SECURITY_AND_PRIVACY.md         # Deployment/data-flow notes
├── .streamlit/
│   └── secrets.toml.example       # Safe configuration template (no real key)
├── src/deductive_ai_coder/
│   ├── models.py                  # Project, codebook, protocol-version models
│   ├── gemini_client.py           # Gemini adapter
│   ├── prompts.py                 # Transparent prompt construction
│   ├── reliability.py             # Agreement calculations
│   ├── data_io.py                 # CSV/Excel/SPSS I/O
│   ├── text_extract.py            # PDF/DOCX/TXT/MD source extraction
│   ├── project_io.py              # Portable project JSON
│   └── audit.py                   # Reproducibility/audit record
└── tests/
    └── test_core.py
```

## Local setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create the local secret file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

On Windows, simply copy the example file manually and rename it to `secrets.toml`.

Add your Gemini key to `.streamlit/secrets.toml`, then run:

```bash
streamlit run app.py
```

The real `.streamlit/secrets.toml` is ignored by Git and must **not** be committed.

## Deployment

The app reads:

- `GEMINI_API_KEY` — required for AI functions
- `GEMINI_MODEL` — defaults to `gemini-3.8-flash`
- `MAX_BATCH_ROWS` — defaults to 1000
- `MAX_CALIBRATION_ROWS` — defaults to 100

These may be configured through Streamlit secrets or equivalent server environment variables. A centrally managed API key means users can consume the owner's API quota, so an open public deployment needs additional access control, quotas, authentication, or a different payment/API-key model. The MVP deliberately does not solve that product-level problem yet.

## What is sent to Gemini?

Only actions that use AI send material to the configured Gemini service.

Depending on the action, that can include:

- project/research context;
- the current codebook and coding instructions;
- literature/source text supplied for protocol co-design;
- calibration case text and optional case context;
- selected disagreement cases for protocol refinement;
- full-dataset text and the case-level context columns explicitly selected by the researcher.

Human-code comparisons used in refinement may also be included in the refinement request.

The application requires a user acknowledgement before AI calls. That acknowledgement is **not** a substitute for ethics approval, data-processing agreements, institutional policy, informed consent, or research-data-management review where those are required.

## What is stored?

### Downloadable project JSON

The project file contains:

- project description;
- codebook draft;
- frozen protocol versions;
- general coding instructions;
- limited calibration summary metrics;
- names of literature/source files used during the session.

It intentionally does **not** store raw participant calibration text, full datasets, or extracted literature text.

### Active Streamlit session

Uploaded/extracted material and coding results may remain in the current server session while the app is in use. The MVP does not implement a persistent database.

### Provider-side processing

The Gemini adapter sends each request with `store=False`, so this MVP does not use the Interactions API's optional server-side conversation storage. Other provider-side processing, abuse monitoring, legal retention, and applicable terms remain governed by the configured Gemini service and account/project. Researchers/deployers must assess whether the configured service is appropriate for their data.

## Reliability/calibration scope

For **single-code nominal coding**, the MVP reports raw agreement and Cohen's kappa for:

- Human 1 vs AI;
- Human 2 vs AI (when used);
- Human 1 vs Human 2 (when used).

For **multi-code coding**, it reports exact-set agreement and mean Jaccard similarity. These metrics are deliberately descriptive; the app does not impose an automatic threshold declaring a protocol "validated."

## Known MVP limitations

- No authentication, user accounts, database, or multi-user project collaboration.
- One Gemini provider/model configuration per deployment.
- No guaranteed deterministic replication: hosted model behavior can change even when a model name remains constant.
- Literature extraction is text-based; scanned PDFs without embedded text need OCR outside the app.
- Uploaded literature is truncated in the co-design prompt after a safety/cost limit.
- No inductive code generation pipeline beyond researcher-directed deductive co-design.
- Multi-code human entry is comma-separated rather than a richer multi-select interface.
- SPSS variable names may be sanitized during export; the app displays the mapping when changes are necessary.
- API/network/provider failures are recorded as errors rather than silently converted into an "uncodable" category.
- Public deployment with a centrally funded key needs abuse/cost controls beyond the row limits included here.

## Transparency and reproducibility

The full prompt-building logic is visible in `src/deductive_ai_coder/prompts.py`. The default provider adapter uses the current Gemini Interactions API with structured JSON output; the model name is server-configurable. The Gemini adapter is isolated in `gemini_client.py`, making later provider substitution possible without redesigning the research workflow.

Each full coding run can export an audit JSON containing the frozen protocol snapshot, protocol version, model name, text/context-column configuration, timestamp, and number of rows processed. The participant dataset itself is not duplicated into that audit file.

## License

No open-source license has been selected in this MVP repository. **Source visibility does not by itself grant reuse rights.** Before public release, choose an explicit license that fits the intended research/product strategy and funder/institutional requirements.

## Prototype robustness notes

- Gemini coding responses are constrained to the exact code IDs in the active frozen protocol using a JSON Schema enum.
- The local validator accepts only exact IDs or conservative wrappers such as `CODE A` / `ID: A`; it never fuzzy-matches labels to IDs.
- Calibration shows visible per-case progress and surfaces API/validation failures in an `AI Error` column rather than failing silently.

