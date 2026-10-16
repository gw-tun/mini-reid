# mini-reid backend

Empty Python 3.12 package. Nothing is installed yet.

## Set up

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

`python --version` should print `Python 3.12.x`.

## What goes here later

| Module | Guide step | Purpose |
|---|---|---|
| `app/model.py` | 3 | Load OSNet-AIN and turn a person crop into a 512-number vector |
| `app/gallery.py` | 5 | Store the vectors of many crops and find the nearest ones |
| `app/api.py` | 6 | HTTP API on port 8090 for the frontend |
| `app/tracker.py` | 7-8 | Link the same person across cameras and over time |
