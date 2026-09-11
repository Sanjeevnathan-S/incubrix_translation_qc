import sys
import os
import psutil
from loguru import logger

def setup_logger(log_file: str = "data/outputs/execution.log"):
    """Configures structured application logs for console and file output."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    return logger

def get_peak_memory_mb() -> float:
    """
    Returns peak CPU RAM usage in MB across Windows, Linux, and macOS.
    """
    if sys.platform == "win32":
        # Returns peak working set memory on Windows
        return round(psutil.Process(os.getpid()).memory_info().peak_wset / (1024 * 1024), 2)
    
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns kilobytes
    scale = (1024 * 1024) if sys.platform == "darwin" else 1024
    return round(maxrss / scale, 2)