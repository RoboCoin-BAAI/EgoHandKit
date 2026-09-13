"""Stable, human-readable names for pipeline inputs."""

from pathlib import Path
from typing import Optional


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def sequence_name_for_input(input_path: Path, explicit_name: Optional[str] = None) -> str:
    """Return a collision-resistant sequence name for a video or frame folder.

    Dataset videos commonly share camera basenames (``left.mp4``/``right.mp4``).
    When a video lives below a ``videos`` directory, include its session directory
    so separate sessions cannot reuse the same frame cache or output directory.
    """
    if explicit_name is not None:
        name = explicit_name.strip()
        if not name or name in {'.', '..'} or Path(name).name != name:
            raise ValueError('--sequence_name must be a non-empty single path component')
        return name

    path = Path(input_path)
    if path.suffix.lower() in VIDEO_EXTENSIONS and path.parent.name.lower() in {'videos', 'video'}:
        session_name = path.parent.parent.name
        if session_name:
            return f'{session_name}_{path.stem}'
    return path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name
