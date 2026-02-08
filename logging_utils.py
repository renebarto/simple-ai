#!/usr/bin/env python3
"""
Logging Utilities Module

This module provides centralized logging setup functionality for the MTBF analysis system.
It ensures consistent logging configuration across all components and scripts.

Author: MTBF Analysis Team
Date: October 2025
"""

import logging
import os

def setup_logging(log_file: str, logger_name: str, log_dir: str = 'logs') -> logging.Logger:
    """
    Setup centralized logging configuration for the entire application.
    
    Args:
        log_file: Filename for the log file (relative to log_dir)
        logger_name: Name for the logger
        log_dir: Directory for log files (default: 'logs')
        
    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Construct full log file path
    log_file_path = os.path.join(log_dir, log_file)
    
    # Create main logger
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d:%(funcName)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False  # Prevent propagation to root logger
    
    return logger
