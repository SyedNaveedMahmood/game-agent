import logging
import colorlog

class ModuleColoredFormatter(colorlog.ColoredFormatter):
    """
    Custom formatter to apply different colors to log messages based on the module name.
    """
    def __init__(self, fmt, datefmt=None, style='%', log_colors=None, module_colors=None):
        super().__init__(fmt, datefmt, style, log_colors)
        self.module_colors = module_colors or {}

    def format(self, record):
        # Get the module (logger name)
        module = record.name

        # Determine if there's a specific color for this module and log level
        if module in self.module_colors:
            # Override the log_colors for specific modules
            original_color = self.log_colors.get(record.levelname, '')
            self.log_colors[record.levelname] = self.module_colors[module].get(record.levelname, original_color)
        
        # Format the record using the parent class
        formatted = super().format(record)
        
        # Restore the original color to avoid affecting other loggers
        if module in self.module_colors:
            self.log_colors[record.levelname] = original_color
        
        return formatted
