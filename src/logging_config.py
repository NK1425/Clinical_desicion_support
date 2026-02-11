"""
Structured Logging Configuration
Uses loguru for JSON-formatted, file-rotated logging with correlation IDs
"""
import sys
import uuid
import time
import functools
from contextvars import ContextVar
from loguru import logger

from .config import settings

# Context variable for request correlation IDs
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get or generate a correlation ID for request tracing."""
    cid = correlation_id.get()
    if not cid:
        cid = uuid.uuid4().hex[:12]
        correlation_id.set(cid)
    return cid


def setup_logging():
    """Configure loguru logging with console and file sinks."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[correlation_id]} | "
        "<level>{message}</level>"
    )

    # Console sink
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
        filter=lambda record: record["extra"].setdefault("correlation_id", ""),
    )

    # File sink with rotation
    try:
        logger.add(
            settings.log_file,
            format=log_format,
            level=settings.log_level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            serialize=True,
            filter=lambda record: record["extra"].setdefault("correlation_id", ""),
        )
    except Exception:
        # If log directory doesn't exist or isn't writable, skip file logging
        pass

    return logger


def get_logger(name: str = "cdss"):
    """Get a logger bound with a module name and correlation ID."""
    return logger.bind(correlation_id=get_correlation_id(), module=name)


def timed(func=None, *, name: str = None):
    """Decorator to log execution time of a function."""
    def decorator(fn):
        label = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log = get_logger("timing")
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.info(f"{label} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.error(f"{label} failed after {elapsed:.3f}s: {e}")
                raise

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            log = get_logger("timing")
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.info(f"{label} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.error(f"{label} failed after {elapsed:.3f}s: {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# Initialize logging on import
setup_logging()
