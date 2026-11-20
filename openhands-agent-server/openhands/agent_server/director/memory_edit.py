"""Changing part of a file of team memory in place.

Team memory is kept outside the working tree, so an agent's file editor
cannot reach it. The change itself is made by that same editor
(:class:`openhands.tools.file_editor.editor.FileEditor`, ``str_replace``)
on a copy of the text: the text to replace must be there exactly once,
and is refused, with the lines it was found at, when it is there more
often.
"""

import tempfile
from pathlib import Path

from openhands.sdk.errors import Invalid
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError


def edited(text: str, old_str: str, new_str: str, name: str = "the file") -> str:
    """``text`` with the one place that reads ``old_str`` reading ``new_str``.

    An empty ``old_str`` adds ``new_str`` at the end.

    Raises:
        Invalid: ``old_str`` is not in the text, or is there more than once.
    """
    if not old_str.strip():
        if not new_str.strip():
            raise Invalid("Say what to add, or which text to replace")
        return f"{text.rstrip()}\n\n{new_str.strip()}\n" if text.strip() else new_str
    with tempfile.TemporaryDirectory(prefix="director-edit-") as folder:
        path = Path(folder) / "memory.md"
        path.write_text(text, encoding="utf-8")
        try:
            FileEditor(workspace_root=folder)(
                command="str_replace",
                path=str(path),
                old_str=old_str,
                new_str=new_str,
            )
        except ToolError as error:
            raise Invalid(str(error).replace(str(path), name)) from error
        return path.read_text(encoding="utf-8")
