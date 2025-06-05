import logging
import sys
from datetime import datetime
from typing import Optional, ClassVar

from colorama import Fore, Style, init

# initialize colorama
init()

class CompilerLogger:
    """A custom logger that supports colors and emojis for different log levels"""
    
    # singleton instance
    _instance: ClassVar[Optional['CompilerLogger']] = None
    
    # emoji mappings for different log levels
    EMOJIS = {
        'DEBUG': '🔍 ',
        'INFO': 'ℹ️ ',
        'WARNING': '⚠️ ',
        'ERROR': '❌ ',
        'CRITICAL': '💥 '
    }
    
    # color mappings for different log levels
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    # calculate the maximum length of level names for alignment
    MAX_LEVEL_LENGTH = max(len(level) for level in EMOJIS.keys())
    
    def __new__(cls, name: str = "Compiler", level: int = logging.INFO) -> 'CompilerLogger':
        """Create or return the singleton instance of the logger"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, name: str = "Compiler", level: int = logging.INFO):
        """Initialize the logger with a name and level if not already initialized"""
        if self._initialized:
            return
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove any existing handlers to prevent duplicate messages
        self.logger.handlers = []
        
        # create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        
        self._initialized = True
    
    def _format_message(self, level: str, message: str, **kwargs) -> str:
        """Format the log message with colors and emojis"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        emoji = self.EMOJIS.get(level, '')
        # add extra space for info and warning levels
        if level in ['INFO', 'WARNING']:
            emoji = f"{emoji} "
        color = self.COLORS.get(level, Fore.WHITE)
        
        # format additional kwargs if present
        extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items()) if kwargs else ''
        if extra_info:
            extra_info = f' [{extra_info}]'
        
        # calculate padding for level alignment
        level_padding = ' ' * (self.MAX_LEVEL_LENGTH - len(level))
        
        return f"{color}{emoji}[{level}]{level_padding} {timestamp} {message}{extra_info}{Style.RESET_ALL}"
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(self._format_message('DEBUG', message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(self._format_message('INFO', message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.logger.warning(self._format_message('WARNING', message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.logger.error(self._format_message('ERROR', message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self.logger.critical(self._format_message('CRITICAL', message, **kwargs))

# create the default logger instance
logger = CompilerLogger()

# example usage:
if __name__ == '__main__':
    # demonstrate singleton behavior
    logger1 = CompilerLogger()
    logger2 = CompilerLogger()
    # should print True
    print(f"Are loggers the same instance? {logger1 is logger2}")
    
    logger.debug("This is a debug message", module="parser")
    logger.info("Compilation started", file="main.c")
    logger.warning("Deprecated feature used", feature="old_parser")
    logger.error("Syntax error in file", line=42, column=10)
    logger.critical("Internal compiler error", error_code=500) 