"""Logging configuration for DocGraph MCP server."""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name: str = __name__, log_dir: str = "./logs", log_file: str = "docgraph.log") -> logging.Logger:
    """
    Set up a logger with rotating file handler and console output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    print(f"📝 Logging initialized - File: {log_path} | Rotation: 10MB/5 files")
    
    # Prevent propagation to root logger (avoids duplicate output from uvicorn)
    logger.propagate = False
    
    # Rotating file handler (10MB max size, keep 5 backup files)
    rotating_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    rotating_handler.setLevel(logging.DEBUG)
    
    # Formatter for file logs
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    rotating_handler.setFormatter(file_formatter)
    
    # Console handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(rotating_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance. If not set up, will set it up automatically.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Create default logger for the application
app_logger = setup_logger("docgraph")
