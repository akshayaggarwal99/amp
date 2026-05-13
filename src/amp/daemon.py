"""
AMP daemon · single persistent process, multi-client HTTP.

Replaces the per-spawn `amp serve` pattern for hosts that need many short-
lived clients (Jarvis CLI backends, heartbeat routines, batch evals).
One Python process, one embedding model loaded once, one sqlite handle ·
N HTTP clients share it over FastMCP's streamable-http transport.

Wire-format: FastMCP's `streamable-http` exposes `/mcp` on the bound port.
Clients connect with the MCP TypeScript / Python SDKs' StreamableHTTP
transport. The transport is already supported by `mcp` package the rest
of AMP depends on.

Lifecycle artifacts (all under ~/.amp/):
    daemon.json   {pid, host, port, url, started_at}
    daemon.log    detached stdout/stderr capture
    daemon.lock   advisory file lock · prevents double-start

The daemon is NOT a system service · it lives for the host shell session.
Calling `amp daemon start` again while one is running is a no-op (logs
the existing pid). `amp daemon stop` sends SIGTERM and waits.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DAEMON_DIR = Path.home() / ".amp"
DAEMON_INFO = DAEMON_DIR / "daemon.json"
DAEMON_LOG = DAEMON_DIR / "daemon.log"
DAEMON_LOCK = DAEMON_DIR / "daemon.lock"
DEFAULT_HOST = "127.0.0.1"


@dataclass
class DaemonInfo:
    """Persisted daemon state · written by start, read by stop/status/clients."""
    pid: int
    host: str
    port: int
    url: str
    started_at: str

    @classmethod
    def load(cls) -> Optional["DaemonInfo"]:
        if not DAEMON_INFO.exists():
            return None
        try:
            return cls(**json.loads(DAEMON_INFO.read_text()))
        except Exception:
            return None

    def save(self) -> None:
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)
        DAEMON_INFO.write_text(json.dumps(asdict(self), indent=2))


# ── public API ──────────────────────────────────────────────────────────


def is_running(info: Optional[DaemonInfo] = None) -> bool:
    """True iff the recorded pid is alive AND the port answers."""
    info = info or DaemonInfo.load()
    if not info:
        return False
    if not _pid_alive(info.pid):
        return False
    return _port_open(info.host, info.port)


def start_daemon(port: int = 0, host: str = DEFAULT_HOST) -> DaemonInfo:
    """
    Start the daemon (detached) and return its info. No-op if already running.

    Pass port=0 to let the OS pick a free port · written into daemon.json so
    clients can read it. Fixed ports are fine for users who want one (just
    pass --port 41873).
    """
    existing = DaemonInfo.load()
    if existing and is_running(existing):
        return existing

    # Clean up stale info / lock from a previous crash.
    _cleanup_files()

    if port == 0:
        port = _pick_free_port(host)

    DAEMON_DIR.mkdir(parents=True, exist_ok=True)

    # Spawn `python -m amp.daemon --serve` detached. We re-exec ourselves
    # rather than starting uvicorn inline so the daemon survives the
    # parent typer command exiting.
    log_file = open(DAEMON_LOG, "a")
    log_file.write(f"\n=== {datetime.now(timezone.utc).isoformat()} · launching ===\n")
    log_file.flush()

    proc = subprocess.Popen(
        [sys.executable, "-m", "amp.daemon", "--serve", "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach · daemon survives our exit
        close_fds=True,
    )

    info = DaemonInfo(
        pid=proc.pid,
        host=host,
        port=port,
        url=f"http://{host}:{port}",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    info.save()

    # Wait up to 10 s for the port to come up · short enough that errors
    # surface fast, long enough to cover FastEmbed's first-time load.
    deadline = time.time() + 10
    while time.time() < deadline:
        if _pid_alive(proc.pid) and _port_open(host, port):
            return info
        if not _pid_alive(proc.pid):
            _cleanup_files()
            raise RuntimeError(
                f"AMP daemon exited during startup · see {DAEMON_LOG} for details"
            )
        time.sleep(0.1)

    # Port never came up · let the caller see what happened.
    raise RuntimeError(
        f"AMP daemon failed to bind {host}:{port} within 10 s · see {DAEMON_LOG}"
    )


def stop_daemon(timeout_s: float = 5.0) -> bool:
    """SIGTERM the recorded pid, wait for exit, clear the lock files."""
    info = DaemonInfo.load()
    if not info or not _pid_alive(info.pid):
        _cleanup_files()
        return False

    try:
        os.kill(info.pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_files()
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pid_alive(info.pid):
            _cleanup_files()
            return True
        time.sleep(0.05)

    # Last resort · SIGKILL.
    try:
        os.kill(info.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _cleanup_files()
    return True


def status() -> dict:
    """Snapshot for `amp daemon status` and for health probes."""
    info = DaemonInfo.load()
    if not info:
        return {"running": False, "reason": "no daemon.json"}
    if not _pid_alive(info.pid):
        return {
            "running": False, "reason": "pid is dead",
            "stale_info": asdict(info),
        }
    if not _port_open(info.host, info.port):
        return {
            "running": False, "reason": "pid alive but port closed",
            "stale_info": asdict(info),
        }
    return {"running": True, **asdict(info)}


# ── internals ───────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def _pick_free_port(host: str) -> int:
    """Bind ephemeral, read assigned port, release. Best-available given that
    we can't pass an open socket to the child process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _cleanup_files() -> None:
    for p in (DAEMON_INFO, DAEMON_LOCK):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ── re-exec entrypoint · `python -m amp.daemon --serve` ─────────────────


def _serve_in_place(host: str, port: int) -> None:
    """Run the FastMCP server in this process (called by the detached child)."""
    # Defer the heavy imports to the child process · keeps `amp daemon
    # start` snappy in the parent typer command.
    from amp.server import mcp, initialize

    # Eager initialize · we want the FastEmbed model in RAM BEFORE the
    # first HTTP client connects.
    initialize(verbose=False)

    # Configure transport bind. FastMCP reads host/port from its settings
    # object which we set at construction time in server.py · we mutate it
    # here for the daemon path.
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


def _main() -> None:
    """`python -m amp.daemon --serve --host H --port P` entrypoint."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true", required=True,
                   help="run the FastMCP server in this process")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()

    if not args.serve:
        sys.exit("amp.daemon is for internal re-exec · use `amp daemon start` instead")

    _serve_in_place(args.host, args.port)


if __name__ == "__main__":
    _main()
