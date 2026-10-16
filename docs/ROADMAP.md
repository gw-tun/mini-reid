# Roadmap: from zero to ReID inside semantic-va

**Final goal:** in semantic-va, a user searches, picks a person, presses **Find
similar person**, and sees where else that person appeared. They can also follow
that person live across all cameras.

You get there in 8 phases. Finish each phase's **Done when** before starting the
next. Show your teacher the **Show your teacher** items at the end of each phase.

| Doc | Use it for |
|---|---|
| [FOUNDATIONS.md](FOUNDATIONS.md) | The knowledge (chapters 1–14) |
| [DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) | Downloading scenes and saving search results |
| [LEARNING-GUIDE.md](LEARNING-GUIDE.md) | Building the ReID parts (steps 0–10) |
| [SEMANTIC-VA-NOTES.md](SEMANTIC-VA-NOTES.md) | How semantic-va works, for integration |

```text
Phase 1  Foundations + run semantic-va
Phase 2  Data: pick cameras, download video, save search results in the DB   ← you are here
Phase 3  ReID model + "Find similar person" (mini-reid)
Phase 4  Measure accuracy
Phase 5  Link people across cameras (offline)
Phase 6  Live tracking across cameras
Phase 7  Integrate into semantic-va
Phase 8  Report (VIP-9)
```

---

## Phase 1. Foundations, and run semantic-va

**Learn:** FOUNDATIONS chapters 1, 2, 3, 5 and 13.

**Do:**
- [ ] Start semantic-va on your Mac (`semantic-va/doc/RUN-MAC.md`)
- [ ] Run 5 searches in the UI that find people. Open a clip and look at its
      tracks and crops.
- [ ] Call the API yourself:
  - [ ] `curl http://127.0.0.1:8080/clips/<id>`, and find `tracks`,
        `capture_ts` and `source_url` in the answer
  - [ ] `curl -o c.jpg http://127.0.0.1:8080/clips/<id>/tracks/<key>/crop`
- [ ] Start mini-reid's frontend: `cd frontend && npm run dev`, then open
      http://localhost:3001
- [ ] Start `docs/LOG.md`, your learning log

**Done when:** you can explain how a video becomes clips, tracks and crops in
semantic-va, and which time field to use for recording time.

---

## Phase 2. Data and database

**Learn:** FOUNDATIONS chapters 7 and 10.

**Do:** [DATA-AND-DATABASE.md](DATA-AND-DATABASE.md), all of it:
- [ ] Find a MEVA site and minute with 2–4 cameras and many people (part 3)
- [ ] Download those videos and convert them to MP4 (part 4)
- [ ] Ask your teacher if "save search result" means what part 5 describes
- [ ] Start mini-reid's database (part 6)
- [ ] Write and run `import_search.py` (part 7)
- [ ] Check the saved data with SQL (part 7d)

**Done when:** your database has one or more searches, and 100+ person tracks
from 2+ cameras with recording times and crop files.

**Show your teacher:** the tables (`\dt`), the counts per camera, and a few crops.

---

## Phase 3. ReID model and "Find similar person"

**Learn:** FOUNDATIONS chapters 6, 8, 11 and 12, and the research document on VIP-5.

**Do:** LEARNING-GUIDE steps 2, 3, 5a and 6:
- [ ] Load OSNet-AIN with ReID weights, and write the manifest
- [ ] Embed every saved crop into `reid_embeddings`
- [ ] The SQL similar search works (DATA-AND-DATABASE part 8)
- [ ] FastAPI `POST /similar` on port 8090. Every call saves into
      `reid_queries` and `reid_matches`.
- [ ] Frontend `/search`: person crops, a **Find similar person** button, results,
      and **same / different / unsure** buttons that save a verdict

**Done when:** you click a person and see candidates from other cameras, and your
verdicts are saved in the database.

---

## Phase 4. Measure accuracy

**Learn:** LEARNING-GUIDE steps 5b–5d and 9a.

**Do:**
- [ ] Label 30+ people seen by 2+ cameras, by marking verdicts in your UI.
      (Or use MEVID labels, if your teacher approves; see DATA-AND-DATABASE part 1.)
- [ ] Rank-1 and mAP for OSNet-AIN. If you can, also run SigLIP on the same crops.
- [ ] Sort failures by cause: small crop, back view, look-alike clothes, ...

**Done when:** you have a results table and the top 3 failure causes.

**Show your teacher:** the table, and 5 failure examples with crops side by side.

---

## Phase 5. Link people across cameras (offline)

**Learn:** FOUNDATIONS chapter 9, and LEARNING-GUIDE step 7.

**Do:**
- [ ] A topology file: which cameras connect, and the travel time range
- [ ] `linker.py`: time gate → camera gate → appearance → threshold + margin →
      Hungarian → `global_entities` + `entity_track_links`
- [ ] Frontend `/person/[id]`: the person's route, ordered by recording time

**Done when:** for labelled people the route is correct, and you've counted false
merges and missed links.

---

## Phase 6. Live tracking across cameras

**Learn:** FOUNDATIONS chapters 4 and 12 (SSE), and LEARNING-GUIDE step 8.

**Do:**
- [ ] Replay the 2–4 downloaded MP4s at real speed, one thread or process per camera
- [ ] Detector + ByteTrack per camera, then embed confirmed tracks and match them
      to active identities
- [ ] Save live tracks (`source = 'live'`) and links in the database
- [ ] `GET /events` (SSE), and a frontend `/live` page that follows one person
- [ ] Measure: crops per second, delay to assign an ID, dropped crops

**Done when:** a person walking from camera A to camera B keeps the same global ID
live.

**Show your teacher:** a screen recording of the live page following one person.

---

## Phase 7. Integrate into semantic-va

**Learn:** FOUNDATIONS chapter 13, SEMANTIC-VA-NOTES "Rules to keep", and
LEARNING-GUIDE step 10.

**Do:**
- [ ] Write `docs/INTEGRATION-PLAN.md` and agree it with your teacher **before**
      coding
- [ ] Create a new branch in semantic-va from `main`, named for your Linear issue
      (VIP-9 suggests `tuntunmyint/vip-9-...`)
- [ ] Add the tables as a new `docker/initdb/NN-reid.sql`, and check how
      `tools/migrate.py` applies changes to existing databases
- [ ] Add an embedding backfill tool for existing crops (under `tools/`)
- [ ] Add the endpoints in `src/api/routes.py`, with lazy torch imports
- [ ] Add the button in `web/src/components/ClipTabs.tsx` `TrackRow` (person,
      has crop)
- [ ] Add tests: run semantic-va's existing tests, and add tests for your code
- [ ] Open a pull request and ask for review

**Done when:** the PR is reviewed and merged, and the button works on the team's
index.

---

## Phase 8. Report (VIP-9)

Use the checklist at the end of LEARNING-GUIDE.md. Post the summary on VIP-9 and
attach the full write-up, as VIP-5 asks. Keep **published** numbers (from papers)
and **measured** numbers (from your footage) in separate columns.

---

## When you're stuck

1. Read the error. The last line usually names the problem.
2. Check the services: is the VPN connected? Is Docker running? Are the API (8080),
   the mini-reid DB (5434) and the frontend (3001) up?
3. Write in `LOG.md` what you tried.
4. Ask with the command, the full error, and what you expected.
