"""
SEFS File System Watcher
Monitors the root directory for file operations (create, modify, rename, delete)
and triggers semantic re-analysis.
"""

import time
import logging
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)

logger = logging.getLogger("sefs.watcher")


class SEFSEventHandler(FileSystemEventHandler):
    """
    Custom file system event handler that detects changes to monitored files
    and triggers semantic reprocessing with debouncing.
    """

    def __init__(self, callback, extensions, debounce_seconds=2.0,
                 folder_manager=None):
        super().__init__()
        self.callback = callback
        self.extensions = extensions
        self.debounce_seconds = debounce_seconds
        self.folder_manager = folder_manager
        self._pending_events = {}
        self._timer = None
        self._lock = threading.Lock()
        self.MARKER_FILE = ".sefs_managed"

    def _is_relevant(self, path: str) -> bool:
        """Check if the file event is relevant (correct extension, not a marker file)."""
        p = Path(path)
        if p.name == self.MARKER_FILE:
            return False
        if p.suffix.lower() not in self.extensions:
            return False
        return True

    def _should_suppress(self) -> bool:
        """Check if events should be suppressed (during our own folder operations)."""
        if self.folder_manager and self.folder_manager.suppress_events:
            return True
        return False

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_suppress():
            return
        if self._is_relevant(event.src_path):
            logger.info(f"File created: {event.src_path}")
            self._queue_event("created", event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_suppress():
            return
        if self._is_relevant(event.src_path):
            logger.debug(f"File modified: {event.src_path}")
            self._queue_event("modified", event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if self._should_suppress():
            return
        if self._is_relevant(event.src_path):
            logger.info(f"File deleted: {event.src_path}")
            self._queue_event("deleted", event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if self._should_suppress():
            return
        if self._is_relevant(event.dest_path):
            logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            self._queue_event("moved", event.dest_path, event.src_path)

    def _queue_event(self, event_type: str, path: str, old_path: str = None):
        """Queue an event with debouncing."""
        with self._lock:
            self._pending_events[path] = {
                "type": event_type,
                "path": path,
                "old_path": old_path,
                "time": time.time(),
            }

            # Reset debounce timer
            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(self.debounce_seconds, self._flush_events)
            self._timer.daemon = True
            self._timer.start()

    def _flush_events(self):
        """Process all pending events after debounce period."""
        with self._lock:
            if not self._pending_events:
                return
            events = dict(self._pending_events)
            self._pending_events.clear()

        logger.info(f"Processing {len(events)} file events after debounce")

        try:
            self.callback(events)
        except Exception as e:
            logger.error(f"Error in event callback: {e}")


class FileWatcher:
    """Watches the root directory for file changes and triggers callbacks."""

    def __init__(self, root_folder: Path, extensions: set,
                 debounce_seconds: float = 2.0, folder_manager=None):
        self.root_folder = Path(root_folder)
        self.extensions = extensions
        self.debounce_seconds = debounce_seconds
        self.folder_manager = folder_manager

        self._observer = None
        self._running = False
        self._callback = None

    def start(self, callback):
        """
        Start watching the root directory.

        Args:
            callback: Function called with dict of events after debounce
        """
        if self._running:
            logger.warning("Watcher already running")
            return

        self._callback = callback
        self.root_folder.mkdir(parents=True, exist_ok=True)

        handler = SEFSEventHandler(
            callback=callback,
            extensions=self.extensions,
            debounce_seconds=self.debounce_seconds,
            folder_manager=self.folder_manager,
        )

        self._observer = Observer()
        self._observer.schedule(handler, str(self.root_folder), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True

        logger.info(f"File watcher started on: {self.root_folder}")
        logger.info(f"Monitoring extensions: {self.extensions}")
        logger.info(f"Debounce: {self.debounce_seconds}s")

    def stop(self):
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("File watcher stopped")

    @property
    def is_running(self):
        return self._running
