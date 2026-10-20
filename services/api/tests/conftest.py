import pathlib
import sys

# Tests import `services.*`, so the repository root must be importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
