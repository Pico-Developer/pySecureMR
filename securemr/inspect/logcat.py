#!/usr/bin/env python3
import argparse
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    VERBOSE = ("\033[37m", "V")
    DEBUG = ("\033[36m", "D")
    INFO = ("\033[32m", "I")
    WARN = ("\033[33m", "W")
    ERROR = ("\033[31m", "E")
    FATAL = ("\033[35m", "F")

    def __init__(self, color_code: str, letter: str) -> None:
        self.color_code = color_code
        self.letter = letter


@dataclass
class LogEntry:
    timestamp: str
    pid: str
    tid: str
    level: LogLevel
    tag: str
    message: str


class LogcatReader:
    RESET_COLOR = "\033[0m"

    def __init__(self, package_name: Optional[str], regex_pattern: Optional[str], device: Optional[str]) -> None:
        self.package_name = package_name
        self.regex_pattern = regex_pattern
        self.device = device
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.should_stop = False
        self._compiled_regex = re.compile(regex_pattern) if regex_pattern else None
        self.read_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None

    def _adb_prefix(self) -> list[str]:
        if self.device:
            return ["adb", "-s", self.device]
        return ["adb"]

    def start(self) -> None:
        subprocess.run(self._adb_prefix() + ["logcat", "-c"], check=False)
        self.read_thread = threading.Thread(target=self._read_logs, daemon=True)
        self.read_thread.start()
        self.process_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.process_thread.start()

    def stop(self) -> None:
        self.should_stop = True
        if self.read_thread:
            self.read_thread.join()
        if self.process_thread:
            self.process_thread.join()

    def _get_package_pid(self) -> Optional[str]:
        if not self.package_name:
            return None
        while not self.should_stop:
            try:
                cmd = self._adb_prefix() + ["shell", "pidof", self.package_name]
                result = subprocess.check_output(cmd, text=True)
                pid = result.strip()
                if pid:
                    print(f"Found {self.package_name} PID: {pid}")
                    return pid
            except subprocess.CalledProcessError:
                print(f"Waiting for {self.package_name} to start...")
                time.sleep(1)
        return None

    def _read_logs(self) -> None:
        cmd = self._adb_prefix() + ["logcat", "-v", "threadtime"]
        if self.package_name:
            pid = self._get_package_pid()
            if pid:
                cmd.extend(["--pid", pid])
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )

        while not self.should_stop:
            if not process.stdout:
                break
            line = process.stdout.readline()
            if line:
                self.log_queue.put(line)

        process.terminate()

    def _process_logs(self) -> None:
        log_pattern = re.compile(
            r"(\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\s+"
            r"(\d+)\s+"
            r"(\d+)\s+"
            r"([VDIWEF])\s+"
            r"([^:]+)\s*:\s*"
            r"(.*)"
        )

        while not self.should_stop:
            try:
                line = self.log_queue.get(timeout=1)
            except queue.Empty:
                continue
            match = log_pattern.match(line)
            if not match:
                continue
            timestamp, pid, tid, level_char, tag, message = match.groups()
            level = next((lvl for lvl in LogLevel if lvl.letter == level_char), None)
            if not level:
                continue
            entry = LogEntry(
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                level=level,
                tag=tag.strip(),
                message=message.strip(),
            )
            if self._compiled_regex and not self._compiled_regex.search(line):
                continue
            self._print_log(entry)

    def _print_log(self, entry: LogEntry) -> None:
        print(
            f"{entry.level.color_code}"
            f"{entry.timestamp} {entry.pid}/{entry.tid} {entry.level.letter}/{entry.tag}: {entry.message}"
            f"{self.RESET_COLOR}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Android logcat viewer")
    parser.add_argument("-p", "--package", help="Package name to filter logs")
    parser.add_argument("-e", "--regex", help="Regex pattern to filter log output")
    parser.add_argument("--device", help="ADB device id")
    args = parser.parse_args()

    if not args.package and not args.regex:
        args.regex = "Secure MR"

    reader = LogcatReader(args.package, args.regex, args.device)

    def signal_handler(signum: int, frame: object) -> None:
        print("\nStopping log reader...")
        reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        reader.start()
        while True:
            signal.pause()
    except KeyboardInterrupt:
        reader.stop()


if __name__ == "__main__":
    main()
