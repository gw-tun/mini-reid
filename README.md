# mini-reid

A small practice project for **person re-identification (ReID)**: finding the
same person across different cameras. It will be merged into
[semantic-va](../semantic-va) later.

Linear: VIP-9 (OSNet-AIN for people), part of VIP-5 (ReID for cross-camera
tracking).

## Goals

1. After a search in semantic-va, a **Find similar person** button shows where
   else that person appears.
2. A live view follows that person across all cameras.
3. Learn ReID properly along the way, and produce the evidence VIP-9 asks for.

## Layout

```text
mini-reid/
  frontend/   Next.js app, port 3001
  backend/    Python 3.12 package, empty for now (API on port 8090 from guide step 6)
  docs/
    ROADMAP.md             start here: the 8 phases from zero to semantic-va
    FOUNDATIONS.md         the knowledge: video, detection, tracking, ReID, SQL, APIs
    DATA-AND-DATABASE.md   download scenes, save search results in the database
    LEARNING-GUIDE.md      hands-on ReID building, steps 0-10
    SEMANTIC-VA-NOTES.md   the parts of semantic-va that ReID connects to
  data/       crops and videos (git-ignored, created in DATA-AND-DATABASE.md)
```

| Service | Port |
|---|---|
| mini-reid frontend | 3001 |
| mini-reid backend API | 8090 |
| mini-reid database (Postgres + pgvector) | 5434 |
| semantic-va web / API / database | 3000 / 8080 / 5433 |

## Run

Frontend:

```bash
cd frontend && npm run dev
```

Then open http://localhost:3001.

Backend (empty for now):

```bash
cd backend && source .venv/bin/activate && python --version
```

## Next step

Open [docs/ROADMAP.md](docs/ROADMAP.md). You're in phase 2.
