# file: utils/csv_logger.py
import os
import csv
from threading import Lock
from typing import Dict, List

class CsvLogger:
    def __init__(self, dir_path: str, file_name: str):
        self.dir_path = dir_path
        self.file_path = os.path.join(dir_path, file_name)
        os.makedirs(self.dir_path, exist_ok=True)
        self._lock = Lock()
        self._header_written = os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0
        self._header_fields: List[str] = []
        if self._header_written:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                r = csv.reader(f)
                self._header_fields = next(r, []) or []

    def log_row(self, row: Dict):
        with self._lock:
            if not self._header_written:
                self._header_fields = list(row.keys())
                self._write_header(self._header_fields)
                self._header_written = True
            else:
                new_fields = [k for k in row.keys() if k not in self._header_fields]
                if new_fields:
                    self._extend_header(new_fields)
            line = [row.get(k, '') for k in self._header_fields]
            with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(line)

    def _write_header(self, fields: List[str]):
        with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(fields)

    def _extend_header(self, new_fields: List[str]):
        old_header = self._header_fields
        new_header = old_header + new_fields
        rows = []
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                r = csv.reader(f)
                _ = next(r, None)  # skip old header
                for row in r:
                    rows.append(row + [''] * len(new_fields))
        with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(new_header)
            w.writerows(rows)
        self._header_fields = new_header
