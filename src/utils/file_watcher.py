"""File watcher for incremental graph updates."""
from pathlib import Path
from typing import Callable, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent
import logging

logger = logging.getLogger(__name__)


class CodeFileHandler(FileSystemEventHandler):
    """Handler for code file changes."""
    
    def __init__(self, on_file_changed: Callable, on_file_created: Callable,
                 on_file_deleted: Callable, on_file_moved: Callable,
                 extensions: Optional[Set[str]] = None):
        """Initialize file handler.
        
        Args:
            on_file_changed: Callback for file modifications.
            on_file_created: Callback for file creation.
            on_file_deleted: Callback for file deletion.
            on_file_moved: Callback for file moves.
            extensions: Set of file extensions to watch.
        """
        self.on_file_changed = on_file_changed
        self.on_file_created = on_file_created
        self.on_file_deleted = on_file_deleted
        self.on_file_moved = on_file_moved
        self.extensions = extensions or {'.py', '.js', '.ts', '.java', '.go', '.rs'}
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if file should be processed.
        """
        return Path(file_path).suffix in self.extensions
    
    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification.
        
        Args:
            event: File modification event.
        """
        if not event.is_directory and self._is_code_file(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            self.on_file_changed(event.src_path)
    
    def on_created(self, event: FileCreatedEvent):
        """Handle file creation.
        
        Args:
            event: File creation event.
        """
        if not event.is_directory and self._is_code_file(event.src_path):
            logger.info(f"File created: {event.src_path}")
            self.on_file_created(event.src_path)
    
    def on_deleted(self, event: FileDeletedEvent):
        """Handle file deletion.
        
        Args:
            event: File deletion event.
        """
        if not event.is_directory and self._is_code_file(event.src_path):
            logger.info(f"File deleted: {event.src_path}")
            self.on_file_deleted(event.src_path)
    
    def on_moved(self, event: FileMovedEvent):
        """Handle file move/rename.
        
        Args:
            event: File move event.
        """
        if not event.is_directory:
            if self._is_code_file(event.src_path):
                logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
                self.on_file_moved(event.src_path, event.dest_path)


class FileWatcher:
    """File watcher for monitoring codebase changes."""
    
    def __init__(self, watch_directory: str,
                 on_file_changed: Optional[Callable] = None,
                 on_file_created: Optional[Callable] = None,
                 on_file_deleted: Optional[Callable] = None,
                 on_file_moved: Optional[Callable] = None,
                 extensions: Optional[Set[str]] = None):
        """Initialize file watcher.
        
        Args:
            watch_directory: Directory to watch.
            on_file_changed: Callback for file modifications.
            on_file_created: Callback for file creation.
            on_file_deleted: Callback for file deletion.
            on_file_moved: Callback for file moves (src, dest).
            extensions: Set of file extensions to watch.
        """
        self.watch_directory = Path(watch_directory)
        self.observer = Observer()
        
        # Default callbacks
        def noop(*args, **kwargs):
            pass
        
        self.on_file_changed = on_file_changed or noop
        self.on_file_created = on_file_created or noop
        self.on_file_deleted = on_file_deleted or noop
        self.on_file_moved = on_file_moved or noop
        
        # Set up event handler
        self.handler = CodeFileHandler(
            on_file_changed=self.on_file_changed,
            on_file_created=self.on_file_created,
            on_file_deleted=self.on_file_deleted,
            on_file_moved=self.on_file_moved,
            extensions=extensions
        )
    
    def start(self):
        """Start watching for file changes."""
        if not self.watch_directory.exists():
            logger.warning(f"Watch directory does not exist: {self.watch_directory}")
            return
        
        self.observer.schedule(
            self.handler,
            str(self.watch_directory),
            recursive=True
        )
        self.observer.start()
        logger.info(f"Started watching: {self.watch_directory}")
    
    def stop(self):
        """Stop watching for file changes."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped file watcher")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

