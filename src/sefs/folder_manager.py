"""
SEFS Folder Manager
Manages OS-level folder structure based on semantic clustering.
Creates, updates, and removes folders, and moves files between semantic directories.
"""

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("sefs.folder_manager")


class FolderManager:
    """Manages the OS-level folder structure that mirrors semantic clusters."""

    def __init__(self, root_folder: Path, folder_prefix: str = "",
                 uncategorized_name: str = "Uncategorized"):
        self.root_folder = Path(root_folder)
        self.folder_prefix = folder_prefix
        self.uncategorized_name = uncategorized_name
        self._managed_folders = set()  # Set of folder names managed by SEFS
        self._file_locations = {}  # original_path -> current_os_path
        self._suppress_events = False  # Flag to suppress watcher during our ops

        # Marker file to identify SEFS-managed folders
        self.MARKER_FILE = ".sefs_managed"

        self._load_existing_managed_folders()

    def _load_existing_managed_folders(self):
        """Scan root for existing SEFS-managed folders."""
        if not self.root_folder.exists():
            return

        for item in self.root_folder.iterdir():
            if item.is_dir():
                marker = item / self.MARKER_FILE
                if marker.exists():
                    self._managed_folders.add(item.name)
                    logger.info(f"Found existing managed folder: {item.name}")

    def _sanitize_folder_name(self, name: str) -> str:
        """Create a valid, clean folder name from a label."""
        # Remove invalid characters for folder names
        invalid_chars = '<>:"/\\|?*'
        clean = name
        for ch in invalid_chars:
            clean = clean.replace(ch, "")
        clean = clean.strip(". ")
        if not clean:
            clean = "Unnamed"
        if self.folder_prefix:
            clean = f"{self.folder_prefix}{clean}"
        return clean[:100]  # Limit length

    @property
    def suppress_events(self):
        return self._suppress_events

    @suppress_events.setter
    def suppress_events(self, value):
        self._suppress_events = value

    def apply_clusters(self, clusters: dict) -> dict:
        """
        Apply semantic clusters to the OS-level folder structure.

        Args:
            clusters: dict of cluster_id -> {"label": str, "files": [original_paths]}

        Returns:
            dict mapping original file paths to new OS paths
        """
        self._suppress_events = True
        moved_files = {}

        try:
            # 1. Map cluster labels to folder names
            new_folders = {}
            for cid, cluster_data in clusters.items():
                folder_name = self._sanitize_folder_name(cluster_data["label"])

                # Handle duplicate folder names
                base_name = folder_name
                counter = 1
                while folder_name in new_folders.values():
                    folder_name = f"{base_name}_{counter}"
                    counter += 1

                new_folders[cid] = folder_name

            # 2. Collect all files that need to stay in root (originals)
            all_cluster_files = set()
            for cid, cluster_data in clusters.items():
                for fp in cluster_data["files"]:
                    all_cluster_files.add(fp)

            # 3. Create new semantic folders and move/copy files
            for cid, folder_name in new_folders.items():
                folder_path = self.root_folder / folder_name
                folder_path.mkdir(exist_ok=True)

                # Write marker file
                marker_path = folder_path / self.MARKER_FILE
                if not marker_path.exists():
                    marker_path.write_text(f"SEFS Managed Folder\nCluster: {cid}\nLabel: {clusters[cid]['label']}")

                self._managed_folders.add(folder_name)

                # Copy files into semantic folder (keep originals in root)
                for file_path in clusters[cid]["files"]:
                    src = Path(file_path)
                    if not src.exists():
                        # Check if file was already moved to a managed folder
                        src = self._find_file_in_managed(src.name)
                        if src is None:
                            logger.warning(f"Source file not found: {file_path}")
                            continue

                    dst = folder_path / src.name

                    # Handle name collisions within folder
                    if dst.exists() and dst.resolve() != src.resolve():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = folder_path / f"{stem}_{counter}{suffix}"
                            counter += 1

                    try:
                        if src.parent.resolve() == folder_path.resolve():
                            # File already in correct folder
                            moved_files[file_path] = str(dst)
                            continue

                        # If file is in root, move it
                        if src.parent.resolve() == self.root_folder.resolve():
                            shutil.move(str(src), str(dst))
                            logger.info(f"Moved: {src.name} -> {folder_name}/")
                        # If file is in another managed folder, move it
                        elif src.parent.name in self._managed_folders:
                            shutil.move(str(src), str(dst))
                            logger.info(f"Moved: {src.name} from {src.parent.name}/ -> {folder_name}/")
                        else:
                            # Copy from external location
                            shutil.copy2(str(src), str(dst))
                            logger.info(f"Copied: {src.name} -> {folder_name}/")

                        moved_files[file_path] = str(dst)

                    except Exception as e:
                        logger.error(f"Error moving {src} to {dst}: {e}")

            # 4. Clean up old managed folders that are no longer needed
            self._cleanup_old_folders(set(new_folders.values()))

            self._file_locations = moved_files
            logger.info(f"Folder structure updated: {len(new_folders)} semantic folders, {len(moved_files)} files organized")

        except Exception as e:
            logger.error(f"Error applying clusters: {e}")
        finally:
            self._suppress_events = False

        return moved_files

    def _find_file_in_managed(self, filename: str) -> Path:
        """Find a file within managed folders."""
        for folder_name in self._managed_folders:
            fp = self.root_folder / folder_name / filename
            if fp.exists():
                return fp
        # Also check root
        fp = self.root_folder / filename
        if fp.exists():
            return fp
        return None

    def _cleanup_old_folders(self, current_folders: set):
        """Remove managed folders that are no longer needed."""
        stale = self._managed_folders - current_folders

        for folder_name in stale:
            folder_path = self.root_folder / folder_name
            if folder_path.exists() and folder_path.is_dir():
                marker = folder_path / self.MARKER_FILE
                if marker.exists():
                    # Move any remaining files back to root
                    for item in folder_path.iterdir():
                        if item.name != self.MARKER_FILE:
                            try:
                                dst = self.root_folder / item.name
                                if dst.exists():
                                    dst = self.root_folder / f"{item.stem}_recovered{item.suffix}"
                                shutil.move(str(item), str(dst))
                                logger.info(f"Recovered: {item.name} to root from stale folder {folder_name}")
                            except Exception as e:
                                logger.error(f"Error recovering {item}: {e}")

                    # Remove the empty managed folder
                    try:
                        shutil.rmtree(str(folder_path))
                        logger.info(f"Removed stale managed folder: {folder_name}")
                    except Exception as e:
                        logger.error(f"Error removing folder {folder_path}: {e}")

        self._managed_folders = current_folders

    def get_all_root_files(self) -> list:
        """Get all supported files in root (not in managed subfolders)."""
        from .config import Config

        files = []
        if not self.root_folder.exists():
            return files

        for item in self.root_folder.iterdir():
            if item.is_file() and item.suffix.lower() in Config.WATCH_EXTENSIONS:
                if item.name != self.MARKER_FILE:
                    files.append(str(item))

        return files

    def get_all_files(self) -> list:
        """Get all supported files in root AND managed subfolders."""
        from .config import Config

        files = []
        if not self.root_folder.exists():
            return files

        for item in self.root_folder.rglob("*"):
            if item.is_file() and item.suffix.lower() in Config.WATCH_EXTENSIONS:
                if item.name != self.MARKER_FILE:
                    files.append(str(item))

        return files

    def get_managed_folders(self) -> list:
        """Return list of currently managed folders with their contents."""
        result = []
        for folder_name in sorted(self._managed_folders):
            folder_path = self.root_folder / folder_name
            if folder_path.exists():
                files_in_folder = []
                for item in folder_path.iterdir():
                    if item.is_file() and item.name != self.MARKER_FILE:
                        files_in_folder.append({
                            "name": item.name,
                            "path": str(item),
                            "size": item.stat().st_size,
                        })
                result.append({
                    "name": folder_name,
                    "path": str(folder_path),
                    "file_count": len(files_in_folder),
                    "files": files_in_folder,
                })
        return result

    def reset(self):
        """Reset: move all files back to root and remove managed folders."""
        self._suppress_events = True
        try:
            for folder_name in list(self._managed_folders):
                folder_path = self.root_folder / folder_name
                if folder_path.exists():
                    for item in folder_path.iterdir():
                        if item.name != self.MARKER_FILE and item.is_file():
                            dst = self.root_folder / item.name
                            if not dst.exists():
                                shutil.move(str(item), str(dst))
                    shutil.rmtree(str(folder_path), ignore_errors=True)

            self._managed_folders.clear()
            self._file_locations.clear()
            logger.info("Folder structure reset: all files restored to root")
        finally:
            self._suppress_events = False
