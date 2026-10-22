import pathlib
import sys

# Tests import `services.worker.app...`, as the worker does: from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
