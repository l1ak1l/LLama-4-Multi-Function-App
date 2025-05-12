import os
import time
from pathlib import Path
from datetime import datetime, timedelta

class TempCleanup:
    @staticmethod
    def cleanup(max_age_hours: int = 24):
        temp_dirs = ["temp/embeddings", "temp/uploads"]
        
        for dir_path in temp_dirs:
            dir = Path(dir_path)
            if dir.exists():
                for file in dir.iterdir():
                    file_age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
                    if file_age > timedelta(hours=max_age_hours):
                        file.unlink()

    @classmethod
    def start_cleanup_scheduler(cls, interval_hours: int = 6):
        while True:
            cls.cleanup()
            time.sleep(interval_hours * 3600)