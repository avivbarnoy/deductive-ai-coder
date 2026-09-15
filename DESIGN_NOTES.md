# Design Notes — MVP Scope and Rationale

## Why this is a rebuild rather than a patch

The earlier proof-of-concept coupled the interface to one study-specific taxonomy. Its Stage 1 could rewrite a prose prompt, while Stages 2 and 3 still relied on a separate hard-coded code dictionary. That made the apparent protocol editable while the actual coding options remained fixed.

This MVP makes one object authoritative: a **structured research protocol/codebook**. The same protocol snapshot generates the AI coding request, defines valid codes, drives human coding options, and is stored in the audit record.

## Current scope

The product question for this version is intentionally narrow:

> Can a researcher develop a deductive qualitative coding protocol with GenAI support, calibrate AI coding against human coding, refine the protocol from disagreements, and then apply a frozen version reproducibly to a larger dataset?

The app therefore prioritizes an end-to-end workflow over a large feature set.

## Stage 1 decisions

- Deductive coding only.
- The researcher can begin with no categories, an initial codebook, or literature/source material.
- AI co-design returns a complete structured draft rather than an opaque replacement prompt.
- Code IDs are strings; the core does not assume `1.1`, `2.3`, or any other numbering scheme.
- Single-code mode is the simplest default; multi-code and optional hierarchy are available without dominating the interface.
- A visible "advanced" view exposes the generated protocol logic for transparency.

## Stage 2 decisions

- One or two human coders.
- AI returns code IDs only during normal calibration.
- Explanations are generated only when a researcher asks about a specific case.
- Agreement statistics are reported, but the software does not impose an automatic validity threshold.
- Disagreement cases can be sent back to Gemini to propose the smallest useful protocol refinements.
- Refinements become a working draft; they do not rewrite a frozen protocol version.

## Stage 3 decisions

- Uses only a frozen protocol version.
- Accepts CSV, Excel, and SPSS SAV.
- The researcher selects the text column and any case-level context columns the AI may use.
- Original columns are preserved and AI outputs are appended.
- API/provider errors remain explicit errors; they are never silently translated into a substantive code.
- Exports include a separate audit JSON containing the exact frozen protocol/configuration used.

## Data and persistence decisions

A downloadable project JSON stores the research protocol, versions, and limited calibration summaries. It intentionally does not store raw participant data or extracted literature text. This keeps the MVP portable while reducing accidental persistence.

A production system may eventually need a proper database, authentication, deletion controls, and institutional storage. Those are deliberately deferred until user testing clarifies the scaling route.

## API architecture

The Gemini implementation lives behind a small `GeminiService` adapter. The rest of the application does not directly call provider SDK functions. This is enough abstraction for a later "bring your own model/key" design without spending MVP development effort on multiple providers now.

The deployment owns the API key. Test users do not type it into the browser UI.

## Explicit non-goals for this MVP

- general inductive/thematic coding;
- NVivo/MAXQDA replacement;
- collaborative coding workspaces;
- unlimited human coders;
- comprehensive inter-rater reliability framework;
- user accounts and billing;
- model marketplace/provider switching;
- automatic claims that a codebook is methodologically "validated."

## What user testing should teach us

The pilot should reveal, among other things:

- whether researchers understand the three-stage workflow;
- how much AI assistance they want during codebook construction;
- whether the structured fields are sufficient or too demanding;
- whether on-demand explanations help resolve disagreements;
- which agreement outputs researchers actually use;
- how often multi-code or hierarchy options are needed;
- what import/export formats matter in practice;
- where users become uncertain about privacy, model behavior, or methodological responsibility;
- what should be built next if the tool moves from proof-of-concept to adaptation/scaling.
