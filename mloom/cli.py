import click
from mloom.core.config import config

@click.group()
def cli():
    """Mloom CLI: The Local-First LLMOps Platform"""
    pass

@cli.group()
def server():
    """Manage server connections (Local SQLite vs Remote Server)"""
    pass

@server.command("connect")
@click.argument("target")
def connect(target: str):
    """
    Connect to a server.
    Example: 
      mloom server connect local
      mloom server connect http://192.168.1.5:8000
    """
    if target.lower() == "local":
        config.mode = "local"
        config._save_config()
        click.secho("[Mloom] Switched to Local SQLite mode.", fg="green")
        click.secho(f"[Mloom] Database path: {config.local_db_path}", fg="blue")
    else:
        if not target.startswith("http"):
            target = f"http://{target}"
            
        config.mode = "remote"
        config.backend_url = target
        config._save_config()
        click.secho(f"[Mloom] Switched to Remote Mode. Target: {target}", fg="green")

if __name__ == "__main__":
    cli()