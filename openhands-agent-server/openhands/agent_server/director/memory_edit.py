"""Changing part of a file of team memory in place.

Team memory is kept outside the working tree, so an agent's file editor
cannot reach it. The change itself is made by that same editor
(:class:`openhands.tools.file_editor.editor.FileEditor`) on a copy of the
text, with the editor's own commands and arguments: ``str_replace``
needs the text to replace to be there exactly once, ``insert`` puts text
after a line.
"""

import tempfile
from pathlib import Path
from typing import Literal

from openhands.sdk.errors import Invalid
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError


EditCommand = Literal["str_replace", "insert"]


def edited(
    text: str,
    command: EditCommand,
    *,
    old_str: str = "",
    new_str: str = "",
    insert_line: int | None = None,
    name: str = "the file",
) -> str:
    """``text`` after one command of the file editor.

    Args:
        insert_line: For ``insert``, the line ``new_str`` goes after; 0 is
            the top. Left out, it goes at the end, which the editor itself
            has no word for.

    Raises:
        Invalid: The editor refuses the change: ``old_str`` is not in the
            text or is there more than once, or there is no such line.
    """
    if command == "insert" and not new_str.strip():
        raise Invalid("Say in `new_str` what to insert")
    if command == "str_replace" and not old_str:
        raise Invalid("Say in `old_str` which text to replace")
    with tempfile.TemporaryDirectory(prefix="director-edit-") as folder:
        path = Path(folder) / "memory.md"
        path.write_text(text, encoding="utf-8")
        editor = FileEditor(workspace_root=folder)
        try:
            if command == "insert":
                if insert_line is None:
                    insert_line = len(text.splitlines())
                    if text.strip():
                        new_str = "\n" + new_str
                editor(
                    command="insert",
                    path=str(path),
                    insert_line=insert_line,
                    new_str=new_str,
                )
            else:
                editor(
                    command="str_replace",
                    path=str(path),
                    old_str=old_str,
                    new_str=new_str,
                )
        except ToolError as error:
            raise Invalid(str(error).replace(str(path), name)) from error
        return path.read_text(encoding="utf-8")
