"""Small terminal-screen model used to read interactive subprocess output."""

from __future__ import annotations

import unicodedata


class TerminalScreen:
    """Interpret the VT100 operations emitted by Rich's live display."""

    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.columns = columns
        self._lines = [[" "] * columns for _ in range(rows)]
        self._row = 0
        self._column = 0
        self._saved_cursor = (0, 0)
        self._pending = ""

    def feed(self, chunk: str) -> None:
        text = self._pending + chunk
        self._pending = ""
        index = 0
        while index < len(text):
            character = text[index]
            if character == "\x1b":
                if index + 1 >= len(text):
                    self._pending = text[index:]
                    break
                if text[index + 1] == "[":
                    end = index + 2
                    while end < len(text) and not ("@" <= text[end] <= "~"):
                        end += 1
                    if end >= len(text):
                        self._pending = text[index:]
                        break
                    self._apply_csi(text[index + 2 : end], text[end])
                    index = end + 1
                    continue
                index += 2
                continue
            if character == "\r":
                self._column = 0
            elif character == "\n":
                self._line_feed()
            elif character == "\b":
                self._column = max(0, self._column - 1)
            elif character >= " ":
                self._write(character)
            index += 1

    def extract_block(self, start_marker: str, end_marker: str) -> str | None:
        """Return the latest screen block delimited by the two markers."""
        lines = ["".join(line).rstrip() for line in self._lines]
        starts = [index for index, line in enumerate(lines) if start_marker in line]
        if not starts:
            return None
        start = starts[-1]
        for end in range(start, len(lines)):
            if end_marker in lines[end]:
                return "\n".join(lines[start : end + 1]).strip()
        return None

    def _write(self, character: str) -> None:
        if unicodedata.combining(character):
            if self._column:
                self._lines[self._row][self._column - 1] += character
            return
        width = 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
        if self._column >= self.columns:
            self._column = 0
            self._line_feed()
        self._lines[self._row][self._column] = character
        if width == 2 and self._column + 1 < self.columns:
            self._lines[self._row][self._column + 1] = ""
        self._column += width

    def _line_feed(self) -> None:
        if self._row == self.rows - 1:
            self._lines.pop(0)
            self._lines.append([" "] * self.columns)
        else:
            self._row += 1

    def _apply_csi(self, parameters: str, command: str) -> None:
        clean_parameters = parameters.lstrip("?")
        values = [int(value) if value else 0 for value in clean_parameters.split(";")] if clean_parameters else []
        amount = values[0] if values and values[0] else 1
        if command == "A":
            self._row = max(0, self._row - amount)
        elif command == "B":
            self._row = min(self.rows - 1, self._row + amount)
        elif command == "C":
            self._column = min(self.columns - 1, self._column + amount)
        elif command == "D":
            self._column = max(0, self._column - amount)
        elif command == "E":
            self._row = min(self.rows - 1, self._row + amount)
            self._column = 0
        elif command == "F":
            self._row = max(0, self._row - amount)
            self._column = 0
        elif command == "G":
            self._column = min(self.columns - 1, max(0, amount - 1))
        elif command in {"H", "f"}:
            row = values[0] if values and values[0] else 1
            column = values[1] if len(values) > 1 and values[1] else 1
            self._row = min(self.rows - 1, max(0, row - 1))
            self._column = min(self.columns - 1, max(0, column - 1))
        elif command == "J":
            self._erase_display(values[0] if values else 0)
        elif command == "K":
            self._erase_line(values[0] if values else 0)
        elif command == "s":
            self._saved_cursor = (self._row, self._column)
        elif command == "u":
            self._row, self._column = self._saved_cursor

    def _erase_line(self, mode: int) -> None:
        if mode == 1:
            start, end = 0, self._column + 1
        elif mode == 2:
            start, end = 0, self.columns
        else:
            start, end = self._column, self.columns
        self._lines[self._row][start:end] = [" "] * (end - start)

    def _erase_display(self, mode: int) -> None:
        if mode in {2, 3}:
            self._lines = [[" "] * self.columns for _ in range(self.rows)]
            return
        if mode == 1:
            for row in range(self._row):
                self._lines[row] = [" "] * self.columns
            self._lines[self._row][: self._column + 1] = [" "] * (self._column + 1)
            return
        self._lines[self._row][self._column :] = [" "] * (self.columns - self._column)
        for row in range(self._row + 1, self.rows):
            self._lines[row] = [" "] * self.columns
