import os
import tempfile
from pathlib import Path

# Point the API at a throwaway SQLite file before api.db is imported.
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"
