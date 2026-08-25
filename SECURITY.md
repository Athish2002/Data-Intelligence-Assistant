# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | ✅ Yes    |
| 1.0.x   | ❌ No     |

## Data Handling Guarantees

- **In-memory only** — uploaded datasets are never written to disk.
- **No telemetry** — no data is sent to any external service by DIA itself.
- **Credential safety** — cloud credentials entered in the UI are stored only in Streamlit session state for the duration of the session and are never logged.
- **PII detection** — the app warns when column names suggest personally identifiable information (email, SSN, phone, etc.), but it is the **user's responsibility** to ensure they have the right to process the data.

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public GitHub issue.

Instead, open a [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability) on this repository so it can be addressed privately.

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to respond within **72 hours** and release a fix within **7 days** for critical issues.

## Known Limitations

- DIA is intended for **development and research use**. Do not expose it directly to the internet without additional authentication.
- The Streamlit session model means that on a shared server, multiple users share the same Python process. Do not deploy to shared infrastructure without proper multi-tenant isolation (e.g., Streamlit Community Cloud isolates sessions per user).
- Optional cloud integrations (S3, GCS, Azure, BigQuery, Snowflake) pass credentials directly to official SDKs and are subject to each provider's security model.
