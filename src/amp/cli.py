import typer
import uvicorn # Not strictly used by FastMCP stdio but good to have deps
from amp.server import mcp, initialize
from amp.core.storage import Storage
import shutil
from pathlib import Path

app = typer.Typer()

# `amp daemon ...` sub-app · single persistent server, multi-client HTTP.
# See docs/JARVIS_INTEGRATION.md for the rationale.
daemon_app = typer.Typer(help="Persistent multi-client daemon (HTTP transport).")
app.add_typer(daemon_app, name="daemon")

@app.command()
def serve(
    transport: str = "stdio",
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable activity logs (stderr).")
):
    """
    Start the AMP Memory MCP Server.
    """
    import sys
    if transport == "stdio":
        sys.stderr.write("Starting AMP Memory Server (stdio mode)...\n")
        sys.stderr.write("Loading local embedding model via FastEmbed (this happens once)...\n")
        
        # Trigger download/load visibly
        initialize(verbose=verbose)
        
        sys.stderr.write("Server Ready. Listening on stdio...\n")
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        typer.echo(f"Unknown transport: {transport}")

@app.command()
def setup():
    """
    Download necessary models and initialize the brain.
    Run this first to see progress bars.
    """
    typer.echo("Initializing AMP Brain...")
    typer.echo("Downloading FastEmbed model (BAAI/bge-small-en-v1.5)...")
    
    # This triggers the download
    initialize()
    
    typer.echo("Setup complete! You can now run 'amp serve'.")

@app.command()
def dashboard(port: int = 8000):
    """
    Start the AMP Memory Dashboard (Web UI).
    """
    from amp.dashboard import start_server
    typer.echo(f"Starting Dashboard on http://localhost:{port}")
    start_server(port)

@app.command()
def reset():
    """
    Wipe the brain (Delete ~/.amp/brain.db).
    """
    home = Path.home()
    db_path = home / ".amp" / "brain.db"
    if db_path.exists():
        typer.confirm(f"Are you sure you want to delete {db_path}?", abort=True)
        db_path.unlink()
        typer.echo("Brain wiped.")
    else:
        typer.echo("Brain is already empty.")

# ── daemon subcommands ──────────────────────────────────────────────────


@daemon_app.command("start")
def daemon_start(
    port: int = typer.Option(0, help="TCP port to bind · 0 picks a free one."),
    host: str = typer.Option("127.0.0.1", help="Interface to bind."),
):
    """Start the AMP daemon in the background. No-op if already running."""
    from amp import daemon
    info = daemon.start_daemon(port=port, host=host)
    typer.echo(f"AMP daemon running · pid={info.pid} · {info.url}")
    typer.echo(f"Connect MCP clients to {info.url}/mcp")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the running daemon, if any."""
    from amp import daemon
    stopped = daemon.stop_daemon()
    typer.echo("stopped" if stopped else "no running daemon")


@daemon_app.command("status")
def daemon_status():
    """Show daemon state."""
    from amp import daemon
    import json as _json
    typer.echo(_json.dumps(daemon.status(), indent=2))


@daemon_app.command("restart")
def daemon_restart(
    port: int = typer.Option(0),
    host: str = typer.Option("127.0.0.1"),
):
    """Stop and start in one shot · convenient after upgrades."""
    from amp import daemon
    daemon.stop_daemon()
    info = daemon.start_daemon(port=port, host=host)
    typer.echo(f"AMP daemon restarted · pid={info.pid} · {info.url}")


def main():
    app()

if __name__ == "__main__":
    main()
