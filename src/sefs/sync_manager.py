"""
SEFS Sync Manager
Coordinates bidirectional synchronisation between the semantic engine,
the OS-level folder structure, and the live UI.
Orchestrates the full pipeline: detect changes -> extract -> analyze -> reorganize -> notify UI.
"""

import logging
import threading
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("sefs.sync_manager")


class SyncManager:
    """
    Central coordinator that orchestrates the SEFS pipeline:
    1. File change detected (by watcher or manual trigger)
    2. Content extraction
    3. Semantic analysis & clustering
    4. OS-level folder reorganization
    5. UI notification via WebSocket
    """

    def __init__(self, config, extractor, semantic_engine,
                 folder_manager, file_watcher, socketio=None):
        self.config = config
        self.extractor = extractor
        self.engine = semantic_engine
        self.folder_manager = folder_manager
        self.watcher = file_watcher
        self.socketio = socketio

        self._lock = threading.Lock()
        self._processing = False
        self._last_sync = None
        self._file_registry = {}  # path -> metadata

    def start(self):
        """Start the sync manager: initial scan + start watcher."""
        logger.info("SyncManager starting...")

        # Initial scan of existing files
        self.full_rescan()

        # Start file watcher
        self.watcher.start(callback=self._on_file_events)

        logger.info("SyncManager is running.")

    def stop(self):
        """Stop the sync manager and file watcher."""
        self.watcher.stop()
        logger.info("SyncManager stopped.")

    def full_rescan(self):
        """Perform a full rescan: extract all files, recompute clusters, reorganize."""
        logger.info("Starting full rescan...")

        with self._lock:
            self._processing = True
            self._notify_ui("status", {"message": "Scanning files...", "processing": True})

            try:
                # 1. Collect all files from root directory (including managed subfolders)
                all_files = self.folder_manager.get_all_files()
                # Also check root directly for new files
                root_files = self.folder_manager.get_all_root_files()
                file_set = set(all_files + root_files)

                if not file_set:
                    logger.info("No files found in monitored directory")
                    self._notify_ui("status", {"message": "No files found", "processing": False})
                    self._processing = False
                    return

                logger.info(f"Found {len(file_set)} files to process")
                self._notify_ui("status", {
                    "message": f"Extracting content from {len(file_set)} files...",
                    "processing": True
                })

                # 2. Extract content from all files
                file_data_list = self.extractor.batch_extract(list(file_set))
                valid_files = [f for f in file_data_list if f.get("content", "").strip()]

                if not valid_files:
                    logger.warning("No files with extractable content")
                    self._notify_ui("status", {
                        "message": "No extractable content found",
                        "processing": False
                    })
                    self._processing = False
                    return

                logger.info(f"Extracted content from {len(valid_files)} files")
                self._notify_ui("status", {
                    "message": f"Computing semantic clusters for {len(valid_files)} files...",
                    "processing": True
                })

                # 3. Semantic analysis & clustering
                clusters = self.engine.process_files(valid_files)

                logger.info(f"Computed {len(clusters)} semantic clusters")

                # 4. Apply clusters to OS folder structure
                self._notify_ui("status", {
                    "message": "Updating folder structure...",
                    "processing": True
                })

                moved_files = self.folder_manager.apply_clusters(clusters)

                # 5. Update engine paths to reflect new file locations
                if moved_files:
                    self.engine.update_paths(moved_files)

                # 6. Update registry and notify UI
                self._update_registry(valid_files, clusters, moved_files)
                self._last_sync = datetime.now().isoformat()

                # 7. Send updated graph to UI
                graph_data = self.engine.get_graph_data()
                folder_data = self.folder_manager.get_managed_folders()

                self._notify_ui("update", {
                    "graph": graph_data,
                    "folders": folder_data,
                    "clusters": self._serialize_clusters(clusters),
                    "file_count": len(valid_files),
                    "cluster_count": len(clusters),
                    "last_sync": self._last_sync,
                })

                self._notify_ui("status", {
                    "message": f"Organized {len(valid_files)} files into {len(clusters)} semantic folders",
                    "processing": False
                })

                logger.info(f"Full rescan complete: {len(valid_files)} files -> {len(clusters)} clusters")

            except Exception as e:
                logger.error(f"Full rescan error: {e}", exc_info=True)
                self._notify_ui("status", {
                    "message": f"Error: {str(e)}",
                    "processing": False
                })
            finally:
                self._processing = False

    def _on_file_events(self, events: dict):
        """
        Callback from file watcher when changes are detected.
        Triggers a full rescan to recompute semantic structure.
        """
        if self._processing:
            logger.info("Already processing, skipping event batch")
            return

        event_summary = []
        for path, event in events.items():
            event_summary.append(f"{event['type']}: {Path(path).name}")
            # Handle deletions - remove from engine
            if event["type"] == "deleted":
                self.engine.remove_file(path)

        logger.info(f"File events detected: {', '.join(event_summary)}")

        self._notify_ui("file_event", {
            "events": [
                {"type": e["type"], "file": Path(e["path"]).name}
                for e in events.values()
            ]
        })

        # Full rescan to recompute everything
        self.full_rescan()

    def _update_registry(self, file_data_list: list, clusters: dict, moved_files: dict = None):
        """Update internal file registry with latest data."""
        self._file_registry.clear()
        for f in file_data_list:
            orig_path = f["path"]
            # Use the new path if file was moved
            actual_path = moved_files.get(orig_path, orig_path) if moved_files else orig_path
            self._file_registry[actual_path] = {
                "name": Path(actual_path).name,
                "path": actual_path,
                "size": f.get("size", 0),
                "modified": f.get("modified", ""),
                "extension": f.get("extension", ""),
                "cluster": self.engine.get_file_cluster(actual_path),
                "has_content": bool(f.get("content", "")),
            }

    def _serialize_clusters(self, clusters: dict) -> list:
        """Serialize clusters for JSON transmission."""
        result = []
        for cid, data in clusters.items():
            result.append({
                "id": cid,
                "label": data["label"],
                "file_count": len(data["files"]),
                "files": [Path(f).name for f in data["files"]],
            })
        return result

    def _notify_ui(self, event_type: str, data: dict):
        """Send a real-time event to the UI via WebSocket."""
        if self.socketio:
            try:
                self.socketio.emit(event_type, data, namespace="/")
            except Exception as e:
                logger.warning(f"WebSocket notification error: {e}")

    def get_status(self) -> dict:
        """Return current system status."""
        return {
            "processing": self._processing,
            "last_sync": self._last_sync,
            "file_count": len(self._file_registry),
            "cluster_count": len(self.engine.get_clusters()),
            "watcher_running": self.watcher.is_running,
            "root_folder": str(self.config.ROOT_FOLDER),
            "managed_folders": len(self.folder_manager.get_managed_folders()),
        }

    def get_graph(self) -> dict:
        """Return current graph data."""
        return self.engine.get_graph_data()

    def get_folders(self) -> list:
        """Return managed folder structure."""
        return self.folder_manager.get_managed_folders()

    def get_clusters_serialized(self) -> list:
        """Return serialized clusters for the UI."""
        return self._serialize_clusters(self.engine.get_clusters())

    def get_file_info(self, file_path: str) -> dict:
        """Return metadata for a specific file."""
        return self._file_registry.get(file_path, {})

    def reset_structure(self):
        """Reset: remove all semantic folders, restore files to root."""
        logger.info("Resetting semantic structure...")
        self.folder_manager.reset()
        self._file_registry.clear()
        self._notify_ui("reset", {"message": "Structure reset"})
        logger.info("Reset complete")

    def manual_rescan(self):
        """Trigger a manual rescan (called from UI)."""
        thread = threading.Thread(target=self.full_rescan, daemon=True)
        thread.start()

    def save_metadata(self):
        """Save current state metadata to disk."""
        metadata = {
            "last_sync": self._last_sync,
            "clusters": self._serialize_clusters(self.engine.get_clusters()),
            "file_count": len(self._file_registry),
        }
        try:
            metadata_path = self.config.METADATA_FILE
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
