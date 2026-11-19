"""Gives an agent a git index of its own in a checkout it shares.

Agents that work in one checkout would otherwise stage into one index, and
a commit by one would take along what another had staged. The agent's
terminal finds the ``git`` written here first on its ``PATH``. Inside the
org's repository it runs the real git on an index that belongs to the agent
alone, in which everything outside the team's directory is held at ``HEAD``
and hidden: changes other teams have not committed are neither shown nor
staged, whatever the agent adds or commits. Anywhere else it is the real git.
"""

import os
import shlex
from collections.abc import Sequence
from pathlib import Path

from openhands.agent_server.director.git import GitError, git


# Commands that make a repository or a checkout, which must not be handed
# an index that belongs to this one.
_UNWRAPPED = "clone|init|worktree|submodule"
# Commands that move HEAD. Two agents running one at a time is what keeps a
# commit from being built on an index that is one commit behind.
_MOVES_HEAD = "commit|merge|pull|cherry-pick|revert|rebase|am|reset|checkout|switch"
# Commands that make commits out of what the agent gives them.
_COMMITS = "commit|cherry-pick|revert|am"
_IN_PROGRESS = (
    "MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG rebase-merge rebase-apply"
)
_LOCK_WAIT_SECONDS = 120
# In the repository's git directory. Whoever moves HEAD of a checkout the
# agents work in holds a lock on it.
LOCK_FILE = "director-git.lock"


def _pathspec(path: str, *, exclude: bool = False) -> str:
    return f":(top,{'exclude,' if exclude else ''}literal){path}"


def _hidden(directory: str, sub_team_directories: Sequence[str]) -> list[list[str]]:
    """Pathspec sets that together name everything that is not the team's."""
    sets = [[_pathspec(sub)] for sub in sub_team_directories]
    if directory:
        sets.append([":(top)", _pathspec(directory, exclude=True)])
    return sets


def git_script(
    real_git: str,
    *,
    common_dir: str,
    conversation_id: str,
    directory: str,
    sub_team_directories: Sequence[str],
    excludes: Path,
    flock: str | None,
    readonly: Sequence[str] = (),
    writable: Sequence[str] = (),
) -> str:
    """A ``git`` that keeps one agent's index apart from everyone else's.

    Args:
        common_dir: The git directory of the org's repository, as
            ``git rev-parse --path-format=absolute --git-common-dir`` gives it.
        directory: The repo-relative directory the agent's team owns; '' for
            the whole repository.
        sub_team_directories: Directories inside it that sub-teams own.
        excludes: The ignore file from :func:`excludes_file`.
        readonly: Globs of the boundary the team may only read.
        writable: Globs of the boundary the team may write; none is all.
        flock: The ``flock`` binary, or None where there is none; commands
            that move HEAD are then not kept apart.
    """
    quote = shlex.quote
    hidden = _hidden(directory, sub_team_directories)
    # Changes with what is hidden, so an index synced for other directories
    # is synced again.
    revision = quote("|".join(" ".join(specs) for specs in hidden))
    hide = [
        line
        for specs in hidden
        for line in (
            f'  "$real" reset -q -- {" ".join(map(quote, specs))} || return 1',
            f'  "$real" ls-files -z -- {" ".join(map(quote, specs))} |'
            ' "$real" update-index -z --skip-worktree --stdin || return 1',
        )
    ]
    # What the boundary's globs keep from the team counts as outside too.
    held = [
        *([[f":(glob){glob}" for glob in readonly]] if readonly else []),
        *(
            [
                [
                    f":(glob){directory}/**" if directory else ":(glob)**",
                    *(f":(exclude,glob){glob}" for glob in writable),
                ]
            ]
            # None named is all of it.
            if writable
            else []
        ),
    ]
    outside = [
        f'  "$real" diff --quiet "$1" "$2" -- {" ".join(map(quote, specs))} || return 0'
        for specs in [*hidden, *held]
    ]
    lock = (
        [
            f'  exec 9>>"$common/{LOCK_FILE}"',
            f"  {quote(flock)} -w {_LOCK_WAIT_SECONDS} 9 ||"
            ' fail "another git command holds the checkout; try again"',
        ]
        if flock
        else ["  :"]
    )
    lines = [
        "#!/bin/sh",
        f"real={quote(real_git)}",
        # Set by this script further up the process tree, or by the caller.
        '[ -n "${GIT_INDEX_FILE-}" ] && exec "$real" "$@"',
        'fail() { printf "git: %s\\n" "$1" >&2; exit 1; }',
        # The checkout named is used like any other.
        'while [ "${1-}" = -C ] && [ $# -ge 2 ]; do',
        '  cd "$2" || exit 128',
        "  shift 2",
        "done",
        "command=",
        "skip=",
        'for arg in "$@"; do',
        '  if [ -n "$skip" ]; then skip=; continue; fi',
        '  case "$arg" in',
        "    -C|--git-dir|--git-dir=*|--work-tree|--work-tree=*|--bare)",
        '      exec "$real" "$@" ;;',
        "    -c|--config-env) skip=1 ;;",
        "    -*) ;;",
        "    *) command=$arg; break ;;",
        "  esac",
        "done",
        'case "$command" in',
        f'  ""|{_UNWRAPPED}) exec "$real" "$@" ;;',
        "esac",
        'common=$("$real" rev-parse --path-format=absolute --git-common-dir'
        ' 2>/dev/null) || exec "$real" "$@"',
        f'[ "$common" = {quote(common_dir)} ] || exec "$real" "$@"',
        'git_dir=$("$real" rev-parse --absolute-git-dir) || exec "$real" "$@"',
        f'GIT_INDEX_FILE="$git_dir/agent-index-{conversation_id}"',
        "export GIT_INDEX_FILE",
        "",
        "sync() {",
        '  head=$("$real" rev-parse -q --verify HEAD) || return 0',
        '  [ -f "$GIT_INDEX_FILE" ] || "$real" read-tree HEAD || return 1',
        f'  stamp="$head "{revision}',
        '  was=$(cat "$GIT_INDEX_FILE.synced" 2>/dev/null)',
        '  [ "$was" = "$stamp" ] && return 0',
        # A folder that comes back to the team is in the index as it was
        # when it was hidden: a commit would undo what was committed since.
        # What was hidden is taken as HEAD has it; nothing else is touched.
        '  top=$("$real" rev-parse --show-toplevel) || return 1',
        '  was_hidden=$("$real" -C "$top" ls-files -t | sed -n "s/^S //p")',
        # Taken out first: an entry kept would keep what git last saw of
        # the file, and a change since might not be seen.
        '  [ -z "$was_hidden" ] || {',
        '    printf "%s\\n" "$was_hidden" |'
        ' "$real" -C "$top" update-index --force-remove --stdin &&',
        '    printf "%s\\n" "$was_hidden" |'
        ' "$real" --literal-pathspecs -C "$top" reset -q HEAD --pathspec-from-file=-',
        "  } || return 1",
        '  "$real" ls-files -z |'
        ' "$real" update-index -z --no-skip-worktree --stdin || return 1',
        *hide,
        '  printf "%s\\n" "$stamp" > "$GIT_INDEX_FILE.synced"',
        "}",
        "",
        "# Whether anything that is not the team's differs between two commits.",
        "outside() {",
        *outside,
        "  return 1",
        "}",
        "",
        'case "$command" in',
        f"  {_MOVES_HEAD})",
        *lock,
        "  ;;",
        "esac",
        "busy=",
        f"for marker in {_IN_PROGRESS}; do",
        '  [ -e "$git_dir/$marker" ] && busy=1',
        "done",
        # A commit built on an index that is behind would undo other teams' work.
        '[ -n "$busy" ] || sync </dev/null >/dev/null 2>&1 ||',
        '  fail "your index could not be brought up to date; try again"',
        'before=$("$real" rev-parse -q --verify HEAD)',
        'on=$("$real" symbolic-ref -q HEAD)',
        'had=$("$real" for-each-ref --format="%(refname)" refs/heads)',
        # A reset that would take away other teams' commits is refused
        # before it runs: run, it would change their files and the index.
        '[ "$command" = reset ] && [ -n "$before" ] && {',
        "  to=; named=; dashes=",
        '  for arg in "$@"; do',
        '    if [ -z "$named" ]; then [ "$arg" = reset ] && named=1; continue; fi',
        # A name after --, a second name, or a first that is no commit is a
        # path: the branch does not move.
        '    [ -n "$dashes" ] && { to=; break; }',
        '    case "$arg" in',
        "      --) dashes=1 ;;",
        "      -*) ;;",
        '      *) [ -z "$to" ] && to=$("$real" rev-parse -q --verify "$arg^{commit}")'
        " && continue; to=; break ;;",
        "    esac",
        "  done",
        '  if [ -n "$to" ] && ! "$real" merge-base --is-ancestor "$before" "$to" \\',
        '      && outside "$to" "$before"; then',
        '    fail "that would take away commits of other teams, so it was not'
        ' done; undo your own commit with git revert"',
        "  fi",
        "}",
        # The lock is not for what git leaves running behind it.
        f'"$real" -c core.excludesFile={quote(str(excludes))} "$@" 9>&-',
        "code=$?",
        'case "$command" in',
        "  checkout|switch)",
        # What is accepted is what the checkout's branch holds, and in a
        # shared checkout the branch is every team's.
        '    now=$("$real" symbolic-ref -q HEAD)',
        '    if [ -n "$on" ] && [ "$now" != "$on" ]; then',
        '      "$real" checkout -q "${on#refs/heads/}" || exit 1',
        # A branch the command made is of no use to anyone; one that was
        # there stays.
        '      printf "%s\n" "$had" | grep -qxF -- "$now" ||',
        '        "$real" branch -q -D "${now#refs/heads/}" 2>/dev/null',
        '      fail "the checkout stays on ${on#refs/heads/}, where your work'
        ' is taken from; commit there"',
        "    fi",
        "  ;;",
        "esac",
        'after=$("$real" rev-parse -q --verify HEAD)',
        'case "$command" in',
        f"  {_COMMITS})",
        # What finishes a merge commits what the merge brought.
        '    if [ -z "$busy" ] && [ -n "$before" ] && [ "$before" != "$after" ] \\',
        '        && outside "$before" "$after"; then',
        '      "$real" reset -q --soft "$before"',
        '      rm -f "$GIT_INDEX_FILE.synced"',
        "      fail \"the commit changed files outside your team's folder,"
        ' so it was undone"',
        "    fi",
        "  ;;",
        # Without a sandbox nothing else keeps them from bringing in other
        # teams' changes, which the team's next commits would undo again.
        "  merge|pull|rebase)",
        '    if [ -n "$before" ] && [ "$before" != "$after" ] \\',
        '        && outside "$before" "$after"; then',
        '      "$real" reset -q --keep "$before"',
        '      rm -f "$GIT_INDEX_FILE.synced"',
        "      fail \"that brought changes to files outside your team's folder,"
        ' so it was undone; the director brings in what other teams did"',
        "    fi",
        "  ;;",
        "esac",
        # The checkout's own index has not seen the commit; without this,
        # whoever uses it sees the committed files as staged to be undone.
        'if [ -n "$before" ] && [ -n "$after" ] && [ "$before" != "$after" ] \\',
        '    && [ -f "$git_dir/index" ] \\',
        '    && ! "$real" diff --quiet "$before" "$after" 2>/dev/null; then',
        # The names are relative to the top of the checkout.
        '  top=$("$real" rev-parse --show-toplevel)',
        '  refused=$("$real" diff --no-relative --name-only -z "$before" "$after" |',
        '    GIT_INDEX_FILE="$git_dir/index" "$real" --literal-pathspecs -C "$top" \\',
        '      reset -q "$after" \\',
        "      --pathspec-from-file=- --pathspec-file-nul 2>&1 >/dev/null) ||",
        # The commit stands; whoever uses the checkout should know why
        # its files look staged.
        '    printf "warning: the checkout\'s own index was not updated: %s\\n" \\',
        '      "$refused" >&2',
        "fi",
        "exit $code",
    ]
    return "\n".join(lines) + "\n"


def _ignore_pattern(path: str) -> str:
    """``path`` as a gitignore pattern that matches only that directory."""
    escaped = "".join(f"\\{char}" if char in "\\*?[]" else char for char in path)
    return f"/{escaped}/"


def excludes_file(
    repo_dir: Path, directory: str, sub_team_directories: Sequence[str]
) -> str:
    """An ignore file that hides untracked files outside a team's directory.

    It stands in for the user's own ignore file, so it begins with that.
    """
    segments = directory.split("/") if directory else []
    lines = [_user_excludes(repo_dir).rstrip("\n")]
    for depth in range(len(segments)):
        inside = "/".join(segments[: depth + 1])
        above = _ignore_pattern("/".join(segments[:depth])) if depth else "/"
        lines += [f"{above}*", f"!{_ignore_pattern(inside)}"]
    lines += [_ignore_pattern(sub) for sub in sub_team_directories]
    return "\n".join(line for line in lines if line) + "\n"


def _user_excludes(repo_dir: Path) -> str:
    try:
        configured = git(
            repo_dir, ["config", "--path", "--get", "core.excludesFile"]
        ).strip()
    except (GitError, OSError):
        configured = ""
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    path = Path(configured) if configured else Path(config_home) / "git" / "ignore"
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""


def rcfile(bin_dir: Path) -> str:
    """What an agent's bash reads at its start so that it finds its ``git``.

    The user's own start file may put directories before it on ``PATH``, so
    this runs that file first.
    """
    return (
        "[ -f ~/.bashrc ] && . ~/.bashrc\n"
        f"PATH={shlex.quote(str(bin_dir))}:$PATH\n"
        "export PATH\n"
    )
