import typer
import uvicorn # Not strictly used by FastMCP stdio but good to have deps
from amp.server import mcp, initialize
from amp.core.storage import Storage
import shutil
from pathlib import Path

app = typer.Typer()

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

def main():
    app()

if __name__ == "__main__":
    main()
