"""System tray launcher for packaged SAM3 REST API server."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import TextIO
import urllib.error
import urllib.request
import webbrowser

import pystray
from PIL import Image, ImageDraw

APP_NAME = "CuvisAI SAM3 Server"
ENV_RELATIVE_PATH = Path("configs") / "sam3-server.env"
SERVER_EXE_NAME = "sam3-rest-api.exe"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CuvisAI" / "SAM3" / "logs"
LOG_FILE = LOG_DIR / "sam3-tray.log"
SERVER_LOG_FILE = LOG_DIR / "sam3-server.log"
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

LOGGER = logging.getLogger("sam3_tray")


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)


def parse_env_file(path: Path, app_directory: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().replace("{app}", str(app_directory))
        if key.startswith("SAM3_"):
            values[key] = value
    return values


def _load_server_config(app_directory: Path) -> dict[str, str]:
    defaults = {"SAM3_HOST": "0.0.0.0", "SAM3_PORT": "8100"}
    defaults.update(parse_env_file(app_directory / ENV_RELATIVE_PATH, app_directory))
    return defaults


def _health_url(config: dict[str, str]) -> str:
    host = config.get("SAM3_HOST", "0.0.0.0")
    host = "127.0.0.1" if host in {"0.0.0.0", "*"} else host
    port = config.get("SAM3_PORT", "8100")
    return f"http://{host}:{port}/api/v1/health"


def _open_file_or_dir(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


def _notify(icon: pystray.Icon, title: str, message: str) -> None:
    try:
        icon.notify(message, title)
    except Exception:
        LOGGER.info("%s: %s", title, message)


def _create_icon_image(app_directory: Path) -> Image.Image:
    icon_path = app_directory / "app_icon.ico"
    if icon_path.exists():
        return Image.open(icon_path)
    image = Image.new("RGBA", (64, 64), (21, 82, 125, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 56, 56), outline=(255, 255, 255, 255), width=4)
    draw.text((20, 22), "S3", fill=(255, 255, 255, 255))
    return image


class ServerManager:
    def __init__(self, app_directory: Path, icon: pystray.Icon) -> None:
        self.app_directory = app_directory
        self.icon = icon
        self.process: subprocess.Popen[str] | None = None
        self.process_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready_notified = False
        self.config = _load_server_config(app_directory)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        with self.process_lock:
            if self.is_running:
                return

            server_exe = self.app_directory / SERVER_EXE_NAME
            if not server_exe.exists():
                _notify(self.icon, APP_NAME, f"Missing executable: {server_exe.name}")
                LOGGER.error("Cannot start server. Missing %s", server_exe)
                return

            self.config = _load_server_config(self.app_directory)
            env = os.environ.copy()
            env.update(self.config)

            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.stop_event.clear()
            self.ready_notified = False

            LOGGER.info("Starting server: %s", server_exe)
            self.process = subprocess.Popen(
                [str(server_exe)],
                cwd=str(self.app_directory),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )

            threading.Thread(
                target=self._pipe_to_log,
                args=(self.process.stdout, "STDOUT"),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._pipe_to_log,
                args=(self.process.stderr, "STDERR"),
                daemon=True,
            ).start()
            threading.Thread(target=self._poll_health, daemon=True).start()

            _notify(self.icon, APP_NAME, "SAM3 server starting...")

    def stop(self) -> None:
        with self.process_lock:
            if self.process is None:
                return

            LOGGER.info("Stopping server process")
            self.stop_event.set()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("Server did not exit after terminate(); killing.")
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None

            _notify(self.icon, APP_NAME, "SAM3 server stopped.")

    def open_browser(self) -> None:
        url = _health_url(self.config).replace("/api/v1/health", "/docs")
        LOGGER.info("Opening browser: %s", url)
        webbrowser.open(url)

    def open_logs(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _open_file_or_dir(LOG_DIR)

    def _pipe_to_log(self, stream: TextIO | None, prefix: str) -> None:
        if stream is None:
            return
        handler = RotatingFileHandler(
            SERVER_LOG_FILE,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(message)s]"))
        logger = logging.getLogger(f"sam3_server_pipe_{prefix.lower()}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
        try:
            for line in iter(stream.readline, ""):
                logger.info("%s %s", prefix, line.rstrip())
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _poll_health(self) -> None:
        url = _health_url(self.config)
        while not self.stop_event.is_set() and self.is_running:
            if self._is_healthy(url):
                if not self.ready_notified:
                    self.ready_notified = True
                    _notify(self.icon, APP_NAME, f"SAM3 server is ready at {url}")
                    LOGGER.info("Health check passed: %s", url)
                return
            time.sleep(2)

    @staticmethod
    def _is_healthy(url: str) -> bool:
        try:
            request = urllib.request.Request(url=url, method="GET")
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status == 200
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return False


def _run() -> None:
    app_directory = _app_dir()
    _setup_logging()
    LOGGER.info("Starting tray launcher from %s", app_directory)

    icon = pystray.Icon("sam3-tray", _create_icon_image(app_directory), APP_NAME)
    manager = ServerManager(app_directory, icon)

    def on_start(_icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        manager.start()

    def on_stop(_icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        manager.stop()

    def on_open_browser(_icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        manager.open_browser()

    def on_view_logs(_icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        manager.open_logs()

    def on_exit(_icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        manager.stop()
        _icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem(
            "Start Server",
            on_start,
            enabled=lambda _item: not manager.is_running,
        ),
        pystray.MenuItem(
            "Stop Server",
            on_stop,
            enabled=lambda _item: manager.is_running,
        ),
        pystray.MenuItem("Open in Browser", on_open_browser),
        pystray.MenuItem("View Logs", on_view_logs),
        pystray.MenuItem("Exit", on_exit),
    )
    icon.run()


if __name__ == "__main__":
    _run()

