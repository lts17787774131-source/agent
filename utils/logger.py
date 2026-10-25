# utils/logger.py
# ============================================================
# 企业级日志系统：统一日志格式、分级输出、文件持久化
# ============================================================
import logging
import os
import sys
from typing import Optional

from config import config

# 日志格式
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(log_level: Optional[str] = None, log_to_file: bool = True) -> None:
    """
    初始化全局日志配置。
    幂等设计，多次调用只会初始化一次。
    """
    global _initialized
    if _initialized:
        return

    level = (log_level or config.logging.log_level).upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 清除已有 handler，避免重复输出
    root_logger.handlers.clear()

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # 文件输出
    if log_to_file and config.logging.enable_chat_history:
        log_dir = config.get_abs_path(config.logging.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "agent.log"),
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str = "agent") -> logging.Logger:
    """获取命名日志器（自动初始化日志系统）"""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
