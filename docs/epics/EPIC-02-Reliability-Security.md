# RKB-EPIC-02: RAG Reliability & Security

**Status: ⚠️ Partially complete — core reliability and file-safety work is
done and tested; secrets management and malware scanning are explicitly
flagged as not started.**

**DRI:** John Paul Gummapu (AI Data Engineer) · **Sign-off:** Amith K A
**Depends on:** RAG Platform Foundation (Epic 01)

---

## What this epic is, in plain terms

Epic 01 built a system that works correctly when everything goes right. This
epic is about what happens when things go *wrong* — a server crashes
mid-upload, someone tries to feed it a malicious file, or a stored password
gets exposed. Two stories, both about making the platform trustworthy enough
to run with real data.

---

## Story RKB-STORY-04: Reliability & recovery

**Status: ✅ Done and tested.**

### The problem this solves, in plain terms

Every document upload becomes a background "job" that a worker process picks
up and processes in stages (read → chunk → extract metadata → search-index
it). Before this work, if the worker process crashed or was killed while it
was in the middle of one of those stages, that job would get stuck forever —
marked "in progress" with nothing ever coming back to check on it again. A
customer's document would just silently never finish.

### What was built

- Every job now gets a **time limit ("lease")** the moment a worker starts
  working on it — 5 minutes by default, configurable.
- A background **"reaper"** process checks, once per work cycle, for any job
  whose time limit has expired while still marked "in progress." When it
  finds one, it automatically puts the job back in the queue to be retried
  (or marks it permanently failed if it's already been retried too many
  times) — no human has to notice and manually fix it.
- This check is deliberately isolated so that if the check itself fails (say,
  a brief database hiccup), it can't bring down the whole worker — the
  worker just keeps processing other documents and tries the check again
  next cycle.
- Retry/failure handling and duplicate-upload protection (uploading the
  exact same file twice doesn't create two copies) already existed and
  continue to work as before.

### What's verified

7 dedicated automated tests, including: a simulated crashed worker's job gets
correctly reclaimed; a job that's already failed the maximum allowed number
of times gets permanently marked failed instead of retried forever; a job
still legitimately in progress is never touched; and the worker survives the
reaper itself failing.

---

## Story RKB-STORY-05: Security & file safety

**Status: ⚠️ Partially done.**

### Token hashing — ✅ Done and tested

**The problem, in plain terms:** every user's login credential (their "API
token," the secret string used to prove who they are on every request) was
being stored in the database in **plain, readable text**. Anyone who got
read access to the database — a backup file, a misconfigured permission, an
insider — could read out every user's working credential directly, no
cracking required.

**What was built:** tokens are now run through a one-way scrambling function
(a cryptographic hash) before being stored, and the same scrambling is
applied to whatever token a user presents on each request before comparing
it — so the comparison still works, but the actual usable token is never
sitting in the database in a form anyone could read or reuse. Because the
real token can no longer be recovered from storage, logging back in now
issues a **fresh token** each time (and immediately invalidates the old one)
rather than trying to hand back the original, which is no longer possible
once it's hashed.

**Verified:** a dedicated test confirms the token really isn't stored in
readable form (it checks the database row directly), plus tests confirming
login still works end-to-end and that a token replaced by a new login can no
longer be used.

### File safety validation — ✅ Done and tested

**The problem, in plain terms:** the system would accept and try to process
*any* uploaded file with no guard rails — a specially crafted file could be
small on disk but expand into something enormous once opened (a
"decompression bomb"), or contain thousands of pages, or embed an image with
an absurd resolution — any of which could make the processing worker run out
of memory or hang indefinitely, taking down document processing for
everyone.

**What was built:**
- A cap on how many pages a PDF may have.
- A cap on image resolution, checked *before* the image is fully decoded (so
  a malicious image can't exhaust memory just by being opened) — this
  applies to standalone images and to images embedded inside Word/Excel
  files.
- A cap on how many rows a table (CSV, HTML table, spreadsheet sheet) may
  have.
- A safety time-limit on the document-reading step specifically (the step
  that touches untrusted, attacker-controlled file bytes) — if reading a file
  takes too long, that job is abandoned and marked failed rather than
  hanging the worker forever. This time-limit was deliberately scoped to
  *only* the file-reading step, not the later steps that write to the
  database — abandoning a database write mid-flight is a real correctness
  risk, whereas abandoning a hung file-read is safe.

Every limit above is a configuration value, not a hard-coded constant, so it
can be tuned per deployment without a code change.

**Verified:** 14 dedicated automated tests, including real oversized-PDF,
oversized-image, and oversized-table fixtures that must be rejected, and
confirmation that files *within* every limit are still processed normally.

### Flagged as pending — not started

- **Secrets management.** Right now, secrets (the LiteLLM gateway API key,
  the database connection string) live only in a `.env` file on the server.
  There is no integration with a secrets vault (Azure Key Vault, AWS Secrets
  Manager, HashiCorp Vault, etc.). This is acceptable for local development
  but not for any shared or production environment. **Flagging this now
  because the team member(s) responsible for platform secrets/security
  infrastructure haven't been onboarded to this project yet** — this needs a
  decision on which secrets platform to target before any code is written.
- **Malware scanning.** Uploaded files are not scanned for malware today.
  There is no antivirus/malware-scanning step in the ingestion pipeline.
  **Flagging this for the same reason** — this is a security-operations
  capability that depends on infrastructure/tooling decisions (which scanning
  service, where in the pipeline it runs) that belong to the security team,
  not something to guess at inside the ingestion codebase.

---

## Bottom line for this epic

The two reliability/safety problems that were fully within this codebase's
control — jobs getting stuck forever, and unsafe files crashing the
worker — are fixed and tested. The two remaining items — a real secrets
vault and malware scanning — are genuine gaps, explicitly flagged rather than
worked around, pending the right team members being onboarded to make those
infrastructure calls.
