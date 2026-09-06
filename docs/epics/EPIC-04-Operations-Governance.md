# RKB-EPIC-04: RAG Operations & Governance

**Status: ⏳ Largely not started. This entire epic is flagged as pending —
the team members responsible for observability infrastructure (Prometheus/
Grafana/OpenTelemetry), platform administration tooling, and
compliance/governance (PII, retention policy) have not yet been onboarded to
this project. What exists today is the groundwork these capabilities will be
built on top of, not the capabilities themselves.**

**DRI:** John Paul Gummapu (AI Data Engineer) · **Sign-off:** Amith K A
**Depends on:** RAG Platform Foundation (Epic 01)

---

## What this epic is, in plain terms

The first three epics are about the product working correctly. This epic is
about running it responsibly at an organizational level: can operations
staff see what's happening and get alerted when something's wrong; can an
administrator manage tenants/users/stuck jobs without touching the database
by hand; and can the company meet its data-governance obligations (knowing
what personal information is stored, being able to delete it, keeping proper
records). **None of the three stories in this epic are complete.** This
document exists to make that explicit and easy to hand off, not to claim
progress that hasn't happened.

---

## Story RKB-STORY-08: Monitoring & operations

**Status: ⏳ Not started.**

| Item | Status |
|---|---|
| Prometheus/Grafana monitoring | ⏳ Not started |
| OpenTelemetry tracing | ⏳ Not started |
| Alerting and operational dashboards | ⏳ Not started |

### What exists today (the groundwork, not the capability)

Every part of the system already writes structured logs (machine-readable
JSON, not free-text) with a shared "request ID"/"job ID" attached to every
log line for a given upload or question, and there's a basic `/metrics`
endpoint that returns simple counters (how many uploads, how many queries,
how many failures) from a database table. This means the *plumbing* needed
to eventually feed Prometheus (metrics) and OpenTelemetry (distributed
tracing) already exists in a compatible shape — but it is not connected to
either system today. There is no Grafana dashboard, no alerting rule, no
trace exporter. **Flagging this as fully pending, awaiting the team
member(s) who own observability infrastructure.**

---

## Story RKB-STORY-09: Platform management & deployment

**Status: ⏳ Mostly not started; one sub-item partially exists.**

| Item | Status |
|---|---|
| Admin APIs | ⏳ Not started |
| Job management and DLQ reprocessing | ⚠️ Partial |
| Database migrations | ⏳ Not started |
| Deployment automation | ⚠️ Partial |

### Admin APIs — not started

There is no dedicated admin interface for managing tenants, users, or API
tokens. Today, creating a tenant/admin account is done by running a
command-line script directly on the server, or through a self-service
sign-up page (which provisions one tenant/admin at a time, for that user
only) — there's no admin panel for an operator to manage *other* tenants'
accounts, rotate a user's token, or audit activity across the platform.

### Job management and DLQ reprocessing — partial

**In plain terms, a "DLQ" (dead-letter queue) is where jobs end up after
failing too many times** — they need a human to look at them. Today: a job
that repeatedly fails does get correctly marked "dead" and stops retrying
(this part works and is tested), and there is a manual "reprocess this
document" API call available. What's **missing** is any admin-facing
tooling to *see* the list of dead-lettered jobs, understand why they failed,
or bulk-reprocess them — that has to be done by querying the database
directly today.

### Database migrations — not started

Every schema change so far has been done with hand-written, ad-hoc SQL
(`ALTER TABLE` statements applied at startup) rather than a proper
migrations framework (like Alembic). This works while the schema is changing
frequently in early development, but is a real risk once there's live
production data — retrofitting a migrations framework after that point is
much harder than adopting one now. **Recommended as a near-term priority
given how much schema change has already happened.**

### Deployment automation — partial

There **is** a working, self-contained deployment setup: a Docker Compose
file that brings up the database, cache, API, and worker together with no
manual configuration required, plus a Dockerfile for building the
application image. What's missing is a real CI/CD pipeline (automated
checks that run on every code change before it's allowed to merge or
deploy) and any infrastructure-as-code for cloud deployment.

---

## Story RKB-STORY-10: Governance & compliance

**Status: ⏳ Not started.**

| Item | Status |
|---|---|
| Document versioning | ⏳ Not started |
| PII detection/redaction | ⏳ Not started |
| Retention policies and compliance controls | ⏳ Not started |

### What this means in plain terms

- **Document versioning**: if the same document is re-processed (say, after
  a chunking-quality improvement), the old version's data is simply replaced
  — there's no history kept of prior versions. Also, deleted-and-replaced
  data isn't cleaned up in the background yet, so storage grows over time
  with repeated processing.
- **PII detection/redaction**: the system does not scan ingested documents
  for personally identifiable information (names, SSNs, emails, etc.) or
  redact it. Whatever is in an uploaded document is stored and searchable
  as-is.
- **Retention policies**: there is no automated data-retention or
  deletion-on-schedule capability — data persists until someone manually
  deletes it via the API.

**This entire story is explicitly flagged as not started, pending onboarding
of the team member(s) responsible for governance and compliance
requirements** (legal/compliance sign-off on what "PII detection" and
"retention" need to mean for this specific platform is a prerequisite to
building it correctly — this isn't a decision the engineering team should
make unilaterally).

---

## Bottom line for this epic

This is the one epic where the honest status is: **not built yet, on
purpose, because the people who need to make the calls here haven't joined
the project.** The good news is that nothing here blocks the platform from
being useful and safe to demo or pilot internally today (Epics 01–03 cover
that) — this epic is what's required before it's a fully governed,
observable, admin-managed **production platform**, and it should be treated
as its own follow-on effort once Monitoring, Platform Ops, and
Compliance/Governance owners are in place.
