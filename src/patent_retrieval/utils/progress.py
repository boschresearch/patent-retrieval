# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import copy
from collections import defaultdict

import pandas as pd
from rich.console import Console
from rich.table import Table


class RichTableProgress:
    def __init__(self, total: int | None = None, print_every: int = 1):
        self.total = total
        self.print_every = print_every
        self.i = 0
        self.start_time = pd.Timestamp.now()
        self.console = Console()

    def update(
        self,
        n: int = 1,
        data: dict | None = None,
        sort: bool = False,
        add_defaults: bool = True,
        _default_group: str = "default",
    ) -> None:
        self.i += n

        if data is None:
            data = {}

        elapsed_time = (pd.Timestamp.now() - self.start_time).total_seconds()

        data = copy.deepcopy(data)
        for k in data:
            v = data[k]
            data[k] = (
                v
                if not isinstance(v, RichTableProgress.AvgPerSec)
                else v.resolve(data, elapsed_time)
            )

        if sort:
            data = dict(sorted(data.items(), key=lambda x: x[0]))

        if add_defaults:
            if self.total is not None:
                data = {
                    f"{_default_group}/progress": f"{self.i} / {self.total} ({self.i/self.total*100:.1f}%)",
                    f"{_default_group}/time": self.format_time(elapsed_time),
                    f"{_default_group}/total_time": (
                        self.format_time((elapsed_time / self.i) * self.total)
                        if self.i > 0
                        else "N/A"
                    ),
                    f"{_default_group}/remaining_time": (
                        self.format_time((elapsed_time / self.i) * self.total - elapsed_time)
                        if self.i > 0
                        else "N/A"
                    ),
                    f"{_default_group}/speed": f"{self.i / elapsed_time:.2f} it/s",
                    **data,
                }
            else:
                data = {
                    f"{_default_group}/progress": str(self.i),
                    f"{_default_group}/time": f"{elapsed_time:.2f} sec",
                    f"{_default_group}/speed": f"{self.i / elapsed_time:.2f} it/s",
                    **data,
                }

        if n == 0 or self.i % self.print_every == 0:
            groups = self._group(data)
            if len(groups) > 1:
                table = Table()
                table.add_column("Group")
                table.add_column("Key")
                table.add_column("Value")
                for group, rows in groups.items():
                    middle_i = len(rows) // 2
                    for i, (k, v) in enumerate(rows):
                        g = group if i == middle_i else ""
                        table.add_row(g, str(k), str(v))
                    table.add_section()
            else:
                table = Table()
                table.add_column("Key")
                table.add_column("Value")
                for _, rows in groups.items():
                    for k, v in rows:
                        table.add_row(str(k), str(v))
            self.table = table
            self.print()

    @staticmethod
    def _group(data: dict) -> dict:
        groups = defaultdict(list)
        for k, v in data.items():
            if "/" in k:
                si = k.index("/")
                group = k[:si]
                k = k[si + 1 :]
            else:
                group = ""
            groups[group].append((k, v))
        return dict(groups)

    def print(self) -> None:
        self.console.print(self.table)

    @staticmethod
    def format_time(seconds):
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}min {seconds % 60:.0f}s"
        elif seconds < 86400:
            return f"{seconds // 3600:.0f}h {seconds % 3600 // 60:.0f}min"
        else:
            return f"{seconds // 86400:.0f}days {seconds % 86400 // 3600:.0f}h"

    class AvgPerSec:
        def __init__(self, key: str, unit: str):
            self.key = key
            self.unit = unit

        def resolve(self, data, elapsed_time):
            total = data.get(self.key)
            if total:
                return f"{total / elapsed_time:.2f} {self.unit}"
            else:
                return "N/A"