# mini-reid documentation

The repository is now organized around the finalized CCTV ReID MVP decision. The
old learning-oriented documents remain in `../archive/legacy-docs/` for history;
these documents describe the implementation target.

| Document | Contents |
|---|---|
| [VIP-9-SIMPLE-GUIDE-MY.md](VIP-9-SIMPLE-GUIDE-MY.md) | Start here: simple Myanmar explanation of the project, person pipeline, current progress, and next steps |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | MVP scope, architecture, models, Record Search, Live Follow, phases, and acceptance criteria |
| [DATA_AND_PERFORMANCE.md](DATA_AND_PERFORMANCE.md) | PostgreSQL/Neon schema, pgvector search, indexing jobs, retention, and runtime performance |
| [High-Level Diagram](05-diagrams/HLD.svg) | Service and data-flow overview |
| [Interactive Briefing](06-briefing/index.html) | Browser briefing for the finalized operating model |
| [VIP-9-PERSON-PLAN.md](VIP-9-PERSON-PLAN.md) | Person pipeline development plan (VIP-9): scope, adapters, issues to fix, milestones M0–M6 |
| [VIP-9-DEV-MY.md](VIP-9-DEV-MY.md) | What the person pipeline does so far, high and low level (Myanmar) |

The final operating decisions are:

- MediaMTX records hard-coded RTSP paths into short segments.
- Every completed segment enters asynchronous nearline Record Search indexing.
- The midnight job handles retries, gap detection, backfill, and re-embedding.
- YOLO detection and local tracking run before representative crops are embedded.
- FastReID MSMT17 SBS R50-IBN is the initial person model; FastReID VeRi SBS R50-IBN is the initial vehicle model.
- Live Follow uses an always-on low-rate sentinel and route-aware, bounded ReID bursts.
- Directed camera transition edges schedule downstream windows; the graph narrows search and does not decide identity.
- T-junction branches may run in parallel, using shared model workers and bounded per-camera queues.
- Model versions and vector spaces stay explicit so future OSNet-AIN comparisons cannot mix embeddings.

Recommended reading order: `VIP-9-SIMPLE-GUIDE-MY.md`, then `VIP-9-DEV-MY.md` and
`VIP-9-PERSON-PLAN.md`. Read `PROJECT_SPEC.md` and `DATA_AND_PERFORMANCE.md` for
the detailed project design, then the interactive briefing.
