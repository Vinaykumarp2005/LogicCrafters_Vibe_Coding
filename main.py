"""
SEFS - Semantic Entropy File System
Main entry point. Initializes all components and starts the system.

Usage:
    python main.py                              # Default: monitors ./monitored_root
    python main.py --root /path/to/folder       # Custom root folder
    python main.py --method transformer         # Use sentence-transformers
    python main.py --port 8080                  # Custom web UI port
"""

import sys
import os
import argparse
import logging
import signal
import threading

# Add src/ directory to path so 'from sefs.xxx import ...' resolves correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sefs.config import Config
from sefs.extractor import ContentExtractor
from sefs.semantic_engine import SemanticEngine
from sefs.folder_manager import FolderManager
from sefs.watcher import FileWatcher
from sefs.sync_manager import SyncManager
from sefs.web_server import create_app


def setup_logging(debug=False):
    """Configure logging for SEFS."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # Reduce noise from libraries
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SEFS - Semantic Entropy File System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --root C:\\Users\\Documents\\MyFiles
  python main.py --method transformer --port 8080
  python main.py --debug
        """
    )

    parser.add_argument("--root", type=str, default=None,
                        help="Root folder to monitor (default: ./monitored_root)")
    parser.add_argument("--method", type=str, choices=["tfidf", "transformer"],
                        default="tfidf",
                        help="Embedding method: tfidf (fast) or transformer (accurate)")
    parser.add_argument("--clusters", type=int, default=0,
                        help="Number of clusters (0 = auto-detect)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Similarity threshold (0.0 - 1.0)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web UI port (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Web UI host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    return parser.parse_args()


def main():
    """Main entry point for SEFS."""

    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("sefs.main")

    # ===== Banner =====
    print(r"""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   ███████╗███████╗███████╗███████╗                    ║
    ║   ██╔════╝██╔════╝██╔════╝██╔════╝                    ║
    ║   ███████╗█████╗  █████╗  ███████╗                    ║
    ║   ╚════██║██╔══╝  ██╔══╝  ╚════██║                    ║
    ║   ███████║███████╗██║     ███████║                    ║
    ║   ╚══════╝╚══════╝╚═╝     ╚══════╝                    ║
    ║                                                       ║
    ║   Semantic Entropy File System                        ║
    ║   Self-Organising File Manager                        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    # ===== Apply config =====
    Config.load_from_file()

    if args.root:
        Config.ROOT_FOLDER = os.path.abspath(args.root)
    Config.EMBEDDING_METHOD = args.method
    Config.NUM_CLUSTERS = args.clusters
    Config.SIMILARITY_THRESHOLD = args.threshold
    Config.PORT = args.port
    Config.HOST = args.host

    root = Config.ensure_root_folder()

    logger.info(f"Root folder: {root}")
    logger.info(f"Embedding method: {Config.EMBEDDING_METHOD}")
    logger.info(f"Similarity threshold: {Config.SIMILARITY_THRESHOLD}")
    logger.info(f"Monitored extensions: {Config.WATCH_EXTENSIONS}")

    # ===== Initialize Components =====
    logger.info("Initializing components...")

    # Content extractor
    extractor = ContentExtractor()

    # Semantic engine
    engine = SemanticEngine(
        method=Config.EMBEDDING_METHOD,
        model_name=Config.TRANSFORMER_MODEL,
        num_clusters=Config.NUM_CLUSTERS,
        min_clusters=Config.MIN_CLUSTERS,
        max_clusters=Config.MAX_CLUSTERS,
        similarity_threshold=Config.SIMILARITY_THRESHOLD,
    )

    # Folder manager
    folder_manager = FolderManager(
        root_folder=root,
        folder_prefix=Config.SEMANTIC_FOLDER_PREFIX,
        uncategorized_name=Config.UNCATEGORIZED_FOLDER,
    )

    # File watcher
    file_watcher = FileWatcher(
        root_folder=root,
        extensions=Config.WATCH_EXTENSIONS,
        debounce_seconds=Config.DEBOUNCE_SECONDS,
        folder_manager=folder_manager,
    )

    # Sync manager (coordinates everything)
    sync_manager = SyncManager(
        config=Config,
        extractor=extractor,
        semantic_engine=engine,
        folder_manager=folder_manager,
        file_watcher=file_watcher,
    )

    # Web server + SocketIO
    app, socketio = create_app(sync_manager, Config)

    # ===== Start Sync Manager (initial scan + watcher) =====
    sync_thread = threading.Thread(target=sync_manager.start, daemon=True)
    sync_thread.start()

    # ===== Shutdown handler =====
    def shutdown(sig, frame):
        logger.info("\nShutting down SEFS...")
        sync_manager.save_metadata()
        sync_manager.stop()
        logger.info("Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # ===== Start Web Server =====
    logger.info(f"Starting web UI at http://{Config.HOST}:{Config.PORT}")
    print(f"\n  🌐 Open your browser at: http://{Config.HOST}:{Config.PORT}")
    print(f"  📂 Monitored folder: {root}")
    print(f"  📊 Embedding method: {Config.EMBEDDING_METHOD}")
    print(f"\n  Press Ctrl+C to stop.\n")

    socketio.run(app, host=Config.HOST, port=Config.PORT,
                 debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
