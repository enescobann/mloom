from pathlib import Path


def display_path(path) -> str:
    """Show paths relative to the working directory, or with the home directory as ~."""
    if not path or path == "?":
        return path or "?"
    p = Path(str(path))
    for base, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
        try:
            return prefix + str(p.relative_to(base))
        except ValueError:
            continue
    return str(path)
