# Security and Privacy Notes (MVP)

This file documents security/privacy decisions visible in the source code. It is not legal advice or a claim of institutional compliance.

## API secret

The Gemini API key is read from Streamlit secrets or a server environment variable. There is intentionally no UI field that asks test users to provide the deployment owner's key.

Do not commit `.streamlit/secrets.toml`. The repository includes only `secrets.toml.example`.

## Research data

AI-enabled actions transmit the relevant research material to the configured external Gemini service. The interface therefore asks the user to acknowledge that the material is appropriate for that processing before an AI call is made.

The project JSON does not include participant text or uploaded literature contents. It stores only protocol/project metadata and limited calibration summaries. This reduces accidental persistence but does not mean the deployment is automatically appropriate for sensitive data.

## Logging

The MVP does not intentionally create an application-level raw-data log. Hosting platforms, reverse proxies, monitoring products, or AI providers can have separate logging/retention behavior that must be reviewed for the actual deployment.

## Public testing risk

A centrally funded API key behind a public unauthenticated app can be used by anyone who can access the URL. Row limits reduce accidental costs but are not a security boundary. A larger/public pilot should add suitable access control, rate limiting, quotas, or authentication.

## File handling

Uploaded files are parsed in memory where practical. SPSS read/write uses temporary server files which are removed after processing. No uploaded dataset is deliberately written to a persistent application database.

## Future hardening candidates

- authentication / pilot invitation control;
- per-user or per-project quotas;
- rate limiting;
- institutional/approved AI gateway;
- explicit retention/deletion controls;
- security headers and deployment review;
- privacy-preserving logging/telemetry for user testing;
- vulnerability/dependency scanning;
- threat model and data-protection review before handling sensitive research data.

## Gemini request storage

The Gemini adapter explicitly uses `store=False` for Interactions API requests. The application does not need Gemini's server-side conversation state, so disabling Interaction-object storage reduces unnecessary provider-side retention. This does not override any separate provider logging, abuse-monitoring, legal, billing, or account-level processing requirements.
