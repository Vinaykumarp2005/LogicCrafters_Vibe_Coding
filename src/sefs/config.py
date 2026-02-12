"""
SEFS Configuration Module
Centralized configuration for the Semantic Entropy File System.
"""

import os
import json
from pathlib import Path


class Config:
    """Configuration manager for SEFS."""

    # --- Paths ---
    # Go up from src/sefs/config.py → src/sefs → src → workspace root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    ROOT_FOLDER = BASE_DIR / "monitored_root"
    CONFIG_FILE = BASE_DIR / "sefs_config.json"
    METADATA_FILE = BASE_DIR / ".sefs_metadata.json"

    # --- Semantic Engine ---
    # Method: 'tfidf' (lightweight) or 'transformer' (more accurate, needs GPU/CPU)
    EMBEDDING_METHOD = "tfidf"
    # Clustering: number of clusters (0 = auto-detect via silhouette score)
    NUM_CLUSTERS = 0
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10
    # Similarity threshold for considering files related (0.0 - 1.0)
    SIMILARITY_THRESHOLD = 0.15

    # --- File Watcher ---
    WATCH_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".csv", ".json", ".log"}
    DEBOUNCE_SECONDS = 2.0  # Wait before processing to batch rapid changes

    # --- Folder Manager ---
    SEMANTIC_FOLDER_PREFIX = ""  # Prefix for auto-generated folders
    UNCATEGORIZED_FOLDER = "Uncategorized"

    # --- Web Server ---
    HOST = "127.0.0.1"
    PORT = 5000
    DEBUG = False

    # --- Transformer model (if using transformer method) ---
    TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

    @classmethod
    def ensure_root_folder(cls):
        """Create the monitored root folder if it doesn't exist."""
        cls.ROOT_FOLDER.mkdir(parents=True, exist_ok=True)
        return cls.ROOT_FOLDER

    @classmethod
    def load_from_file(cls):
        """Load configuration overrides from JSON file."""
        if cls.CONFIG_FILE.exists():
            try:
                with open(cls.CONFIG_FILE, "r") as f:
                    overrides = json.load(f)
                for key, value in overrides.items():
                    key_upper = key.upper()
                    if hasattr(cls, key_upper):
                        if key_upper in ("ROOT_FOLDER",):
                            setattr(cls, key_upper, Path(value))
                        elif key_upper == "WATCH_EXTENSIONS":
                            setattr(cls, key_upper, set(value))
                        else:
                            setattr(cls, key_upper, value)
            except Exception as e:
                print(f"[Config] Warning: Could not load config file: {e}")

    @classmethod
    def save_to_file(cls):
        """Save current configuration to JSON file."""
        config_dict = {
            "root_folder": str(cls.ROOT_FOLDER),
            "embedding_method": cls.EMBEDDING_METHOD,
            "num_clusters": cls.NUM_CLUSTERS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "watch_extensions": list(cls.WATCH_EXTENSIONS),
            "host": cls.HOST,
            "port": cls.PORT,
        }
        with open(cls.CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def to_dict(cls):
        """Return configuration as a dictionary."""
        return {
            "root_folder": str(cls.ROOT_FOLDER),
            "embedding_method": cls.EMBEDDING_METHOD,
            "num_clusters": cls.NUM_CLUSTERS,
            "min_clusters": cls.MIN_CLUSTERS,
            "max_clusters": cls.MAX_CLUSTERS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "watch_extensions": list(cls.WATCH_EXTENSIONS),
            "debounce_seconds": cls.DEBOUNCE_SECONDS,
            "host": cls.HOST,
            "port": cls.PORT,
            "transformer_model": cls.TRANSFORMER_MODEL,
        }
