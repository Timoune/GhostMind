import logging
import structlog
from pathlib import Path


class LoggerManager:

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        # FIX: JSON structured logs go to file only.
        # The StreamHandler is removed so heartbeat ticks and other
        # internal events don't bleed into the terminal REPL output.
        # To watch logs live: tail -f logs/ghostmind.log
        file_handler = logging.FileHandler(self.log_dir / "ghostmind.log")
        file_handler.setLevel(logging.INFO)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[file_handler]
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.stdlib.LoggerFactory(),
        )

    def get_logger(self, module_name: str):
        return structlog.get_logger(module_name)
