from pathlib import Path

def load_prompt(file_path: str) -> str:
    """
    Load a prompt from a given file path.

    Parameters
    ----------
    file_path : str
        Path to the prompt file.

    Returns
    -------
    str
        Content of the prompt file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    return path.read_text(encoding="utf-8").strip()
