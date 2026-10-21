# PostgreSQL migrations

`init/001_extensions.sql` is mounted into the official PostgreSQL initialization
directory and enables `pgvector` and `pgcrypto` on a fresh data volume.

The application schema is versioned under `migrations/`:

- `001_extensions.sql` — required extensions
- `002_core_schema.sql` — cameras, recordings, detections, tracks, embeddings,
  search results, live follow, and camera transitions
- `003_seed_models.sql` — MSMT17 and VeRi FastReID model registry rows and default
  threshold policies

Apply them with:

```powershell
python .\scripts\migrate.py
```

The runner tracks applied filenames in `schema_migrations`. The worker, CLI, and
API all persist to this database; `REID_BACKEND=fastreid` is required because the
gallery column is `vector(2048)`.
The initial 2048-D gallery uses exact cosine search because pgvector HNSW does not
accept `vector(2048)` directly. An optional `halfvec(2048)` expression index can be
added later after validating the installed pgvector version.
