import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# When packaged with PyInstaller --onefile, sys.executable points to the .exe;
# __file__ would point to a temp extraction dir and the config would be lost.
_BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
CONFIG_PATH = _BASE / "annotator_config.json"

_DEFAULT_CATEGORIES = [{"id": 1, "name": "defect"}]


class AppConfig:
    """Persists categories and session state to annotator_config.json."""

    def __init__(self):
        self.categories: List[Dict] = list(_DEFAULT_CATEGORIES)
        self.last_folder: Optional[str] = None
        self.last_index: int = 0

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                cats = data.get("categories")
                if cats:
                    cfg.categories = cats
                cfg.last_folder = data.get("last_folder")
                cfg.last_index = int(data.get("last_index", 0))
            except Exception:
                pass
        return cfg

    def save(self):
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "categories": self.categories,
                    "last_folder": self.last_folder,
                    "last_index": self.last_index,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
