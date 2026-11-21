#!/usr/bin/env python3
"""
GitHub Agent: chat with your GitHub account using Claude + the GitHub REST API.

Auth:   a GitHub personal access token (GITHUB_TOKEN) read from the environment / .env
Brain:  Claude via the Anthropic API (ANTHROPIC_API_KEY)

Read-only tools run automatically. Write tools (create issues, push code, open PRs...)
ask for confirmation first (skip with --yes). Pushing straight to a repo's default branch
always asks you to type the branch name unless --allow-default-branch is given.

Usage:
    python github_agent.py              # interactive chat
    python github_agent.py "list my 5 most recently updated repos"
"""
import argparse
import base64
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

API = "https://api.github.com"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TOOL_OUTPUT = 20_000  # characters sent back to the model per tool call
ALLOW_DEFAULT_BRANCH = False  # set by --allow-default-branch
GITHUB_TOKEN_VALUE = ""       # set in main(); used for real `git push`
GIT_BASE = "https://github.com"  # overridable for tests
MAX_STEPS = 15            # safety cap on tool-use iterations per user message


# --------------------------------------------------------------------------- #
# GitHub client
# --------------------------------------------------------------------------- #
class GitHubError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class GitHub:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "claude-github-agent",
            }
        )

    def request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, f"{API}{path}", timeout=30, **kwargs)
        if not resp.ok:
            try:
                msg = resp.json().get("message", resp.text)
            except ValueError:
                msg = resp.text
            raise GitHubError(f"GitHub API {resp.status_code}: {msg}", status=resp.status_code)
        return resp.json() if resp.content else {}


gh: GitHub  # initialised in main()


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def get_authenticated_user():
    u = gh.request("GET", "/user")
    return {k: u.get(k) for k in ("login", "name", "email", "public_repos", "followers", "html_url")}


def list_repos(visibility: str = "all", limit: int = 20):
    repos = gh.request(
        "GET",
        "/user/repos",
        params={"visibility": visibility, "sort": "updated", "per_page": min(limit, 100)},
    )
    return [
        {
            "full_name": r["full_name"],
            "private": r["private"],
            "description": r["description"],
            "language": r["language"],
            "stars": r["stargazers_count"],
            "open_issues": r["open_issues_count"],
            "updated_at": r["updated_at"],
        }
        for r in repos
    ]


def get_repo(repo: str):
    r = gh.request("GET", f"/repos/{repo}")
    keys = ("full_name", "description", "private", "default_branch", "language",
            "stargazers_count", "forks_count", "open_issues_count", "topics",
            "created_at", "pushed_at", "html_url")
    return {k: r.get(k) for k in keys}


def list_issues(repo: str, state: str = "open", limit: int = 20):
    items = gh.request(
        "GET", f"/repos/{repo}/issues",
        params={"state": state, "per_page": min(limit, 100)},
    )
    return [
        {"number": i["number"], "title": i["title"], "state": i["state"],
         "author": i["user"]["login"], "labels": [l["name"] for l in i["labels"]],
         "comments": i["comments"], "url": i["html_url"]}
        for i in items if "pull_request" not in i  # the issues endpoint also returns PRs
    ]


def list_pull_requests(repo: str, state: str = "open", limit: int = 20):
    prs = gh.request(
        "GET", f"/repos/{repo}/pulls",
        params={"state": state, "per_page": min(limit, 100)},
    )
    return [
        {"number": p["number"], "title": p["title"], "state": p["state"],
         "author": p["user"]["login"], "draft": p["draft"],
         "head": p["head"]["ref"], "base": p["base"]["ref"], "url": p["html_url"]}
        for p in prs
    ]


def list_commits(repo: str, branch: str = "", limit: int = 10):
    params = {"per_page": min(limit, 100)}
    if branch:
        params["sha"] = branch
    commits = gh.request("GET", f"/repos/{repo}/commits", params=params)
    return [
        {"sha": c["sha"][:7], "message": c["commit"]["message"].split("\n")[0],
         "author": c["commit"]["author"]["name"], "date": c["commit"]["author"]["date"]}
        for c in commits
    ]


def read_file(repo: str, path: str, ref: str = ""):
    params = {"ref": ref} if ref else {}
    data = gh.request("GET", f"/repos/{repo}/contents/{path}", params=params)
    if isinstance(data, list):  # directory
        return [{"name": f["name"], "type": f["type"], "size": f["size"]} for f in data]
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data


def search_code(query: str, limit: int = 10):
    res = gh.request("GET", "/search/code", params={"q": query, "per_page": min(limit, 30)})
    return [
        {"repo": i["repository"]["full_name"], "path": i["path"], "url": i["html_url"]}
        for i in res.get("items", [])
    ]


# ---- write tools (require confirmation) ----
def create_issue(repo: str, title: str, body: str = "", labels: list | None = None):
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    i = gh.request("POST", f"/repos/{repo}/issues", json=payload)
    return {"number": i["number"], "url": i["html_url"]}


def comment_on_issue(repo: str, number: int, body: str):
    c = gh.request("POST", f"/repos/{repo}/issues/{number}/comments", json={"body": body})
    return {"url": c["html_url"]}



# ---- push & branch tools ----
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode",
             "dist", "build", ".mypy_cache", ".pytest_cache"}
SECRET_GLOBS = ("*.pem", "*.key", "id_rsa*", "id_ed25519*", "*.p12", "*.pfx",
                "credentials.json", ".npmrc", ".pypirc", ".netrc")
MAX_FILE_BYTES = 5_000_000
MAX_FILES = 300


def _is_secret(filename: str) -> bool:
    if filename == ".env" or (filename.startswith(".env.") and filename not in {".env.example", ".env.sample"}):
        return True
    return any(fnmatch.fnmatch(filename, g) for g in SECRET_GLOBS)


def _clean_path(path: str) -> str:
    path = path.replace("\\", "/").strip("/")
    if not path or ".." in path.split("/"):
        raise GitHubError(f"Invalid repository path: {path!r}")
    return path


def _collect_local_files(local_path: str, dest_dir: str = ""):
    root = Path(local_path).expanduser().resolve()
    if not root.exists():
        raise GitHubError(f"Local path not found: {root}")
    prefix = dest_dir.strip("/")
    files, skipped = [], []
    if root.is_file():
        candidates = [(root, (root.name,))]
    else:
        candidates = [(p, p.relative_to(root).parts) for p in sorted(root.rglob("*")) if p.is_file()]
    for p, parts in candidates:
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue
        if _is_secret(p.name):
            skipped.append(f"{'/'.join(parts)} (looks like a secret)")
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"{'/'.join(parts)} (larger than {MAX_FILE_BYTES // 1_000_000} MB)")
            continue
        repo_path = _clean_path("/".join(([prefix] if prefix else []) + list(parts)))
        files.append((repo_path, p))
    if not files:
        raise GitHubError("No pushable files found at that path.")
    if len(files) > MAX_FILES:
        raise GitHubError(f"{len(files)} files found; the limit is {MAX_FILES}. Push a smaller folder.")
    return files, skipped


def _get_branch_sha(repo: str, branch: str):
    try:
        return gh.request("GET", f"/repos/{repo}/git/ref/heads/{branch}")["object"]["sha"]
    except GitHubError as e:
        if e.status in (404, 409):
            return None
        raise


def _guard_default_branch(repo: str, branch: str, default: str):
    if branch != default or ALLOW_DEFAULT_BRANCH:
        return
    print(f"\n  🚨 This will commit DIRECTLY to the default branch '{default}' of {repo}.")
    if input(f"     Type '{default}' to confirm: ").strip() != default:
        raise GitHubError(
            f"Push to default branch '{default}' was not confirmed. "
            "Use a feature branch and open a pull request instead.")


def _commit_files(repo: str, branch: str, files: list, message: str,
                  create_from: str = "", _guarded: bool = False):
    """files: list of (repo_path, bytes). Creates ONE commit containing all of them."""
    default = gh.request("GET", f"/repos/{repo}")["default_branch"]
    if not _guarded:
        _guard_default_branch(repo, branch, default)

    parent = _get_branch_sha(repo, branch)
    if parent is None:  # branch doesn't exist yet: create it from a base branch
        base = create_from or default
        base_sha = _get_branch_sha(repo, base)
        if base_sha is None:  # completely empty repository: seed it via the contents API
            if branch != default:
                _guard_default_branch(repo, default, default)
            first_path, first_data = files[0]
            gh.request("PUT", f"/repos/{repo}/contents/{first_path}", json={
                "message": message, "branch": default,
                "content": base64.b64encode(first_data).decode()})
            if len(files) == 1:
                return {"repo": repo, "branch": default, "files_pushed": 1,
                        "note": "Repository was empty; initial commit created on the default branch."}
            result = _commit_files(repo, default, files[1:], message, _guarded=True)
            result["files_pushed"] += 1  # include the seed file
            return result
        gh.request("POST", f"/repos/{repo}/git/refs",
                   json={"ref": f"refs/heads/{branch}", "sha": base_sha})
        parent = base_sha

    base_tree = gh.request("GET", f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
    tree = []
    for path, data in files:
        blob = gh.request("POST", f"/repos/{repo}/git/blobs",
                          json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    new_tree = gh.request("POST", f"/repos/{repo}/git/trees",
                          json={"base_tree": base_tree, "tree": tree})
    commit = gh.request("POST", f"/repos/{repo}/git/commits",
                        json={"message": message, "tree": new_tree["sha"], "parents": [parent]})
    gh.request("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", json={"sha": commit["sha"]})
    return {"repo": repo, "branch": branch, "commit": commit["sha"][:7], "files_pushed": len(files),
            "url": f"https://github.com/{repo}/commit/{commit['sha']}"}


def push_files(repo: str, branch: str, message: str, files: list, create_from_branch: str = ""):
    prepared = []
    for f in files:
        path = _clean_path(f["path"])
        if _is_secret(path.split("/")[-1]):
            raise GitHubError(f"Refusing to push a file that looks like a secret: {path}")
        prepared.append((path, f["content"].encode("utf-8")))
    if not prepared:
        raise GitHubError("No files provided.")
    return _commit_files(repo, branch, prepared, message, create_from_branch)


def push_local_path(repo: str, branch: str, message: str, local_path: str,
                    dest_dir: str = "", create_from_branch: str = ""):
    files, skipped = _collect_local_files(local_path, dest_dir)
    result = _commit_files(repo, branch, [(rp, p.read_bytes()) for rp, p in files],
                           message, create_from_branch)
    if skipped:
        result["skipped"] = skipped
    return result


def create_branch(repo: str, branch: str, from_branch: str = ""):
    default = gh.request("GET", f"/repos/{repo}")["default_branch"]
    base = from_branch or default
    sha = _get_branch_sha(repo, base)
    if sha is None:
        raise GitHubError(f"Base branch '{base}' not found (is the repository empty?).")
    gh.request("POST", f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": sha})
    return {"repo": repo, "branch": branch, "created_from": base}


def create_pull_request(repo: str, title: str, head: str, base: str = "",
                        body: str = "", draft: bool = False):
    base = base or gh.request("GET", f"/repos/{repo}")["default_branch"]
    pr = gh.request("POST", f"/repos/{repo}/pulls", json={
        "title": title, "head": head, "base": base, "body": body, "draft": draft})
    return {"number": pr["number"], "url": pr["html_url"]}


def create_repo(name: str, private: bool = True, description: str = ""):
    r = gh.request("POST", "/user/repos",
                   json={"name": name, "private": private, "description": description})
    return {"full_name": r["full_name"], "private": r["private"], "url": r["html_url"]}



# ---- real `git push` of a local repository (keeps commit history) ----
def _redact(text: str) -> str:
    for secret in filter(None, (GITHUB_TOKEN_VALUE,
                                base64.b64encode(f"x-access-token:{GITHUB_TOKEN_VALUE}".encode()).decode()
                                if GITHUB_TOKEN_VALUE else "")):
        text = text.replace(secret, "***")
    return text


def _git_auth_env() -> dict:
    # Pass the credential through environment config so it never lands in .git/config,
    # the remote URL, or the process argument list. Requires git >= 2.31.
    basic = base64.b64encode(f"x-access-token:{GITHUB_TOKEN_VALUE}".encode()).decode()
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return env


def _git(path: str, *args: str, env: dict | None = None, check: bool = True) -> str:
    try:
        proc = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                              env=env, timeout=300)
    except subprocess.TimeoutExpired:
        raise GitHubError(f"git {args[0]} timed out")
    out = _redact((proc.stdout + proc.stderr).strip())
    if check and proc.returncode != 0:
        raise GitHubError(f"git {args[0]} failed: {out}")
    return out


def git_push(repo: str, local_path: str, branch: str = "", remote_branch: str = "",
             commit_message: str = ""):
    path = str(Path(local_path).expanduser().resolve())
    if _git(path, "rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise GitHubError(f"{path} is not a git repository. Use push_local_path for plain folders.")
    current = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    local_branch = branch or current
    if local_branch == "HEAD":
        raise GitHubError("Detached HEAD: check out a branch first.")
    remote_branch = remote_branch or local_branch

    default = gh.request("GET", f"/repos/{repo}")["default_branch"]
    _guard_default_branch(repo, remote_branch, default)

    committed = False
    if commit_message:
        if local_branch != current:
            raise GitHubError(f"Can only commit on the checked-out branch ('{current}').")
        _git(path, "add", "-A")
        staged = _git(path, "diff", "--cached", "--name-only").splitlines()
        bad = [f for f in staged if _is_secret(f.split("/")[-1])]
        if bad:
            _git(path, "reset", "-q")
            raise GitHubError(f"Refusing to commit files that look like secrets: {bad}. "
                              "Add them to .gitignore and try again.")
        if staged:
            _git(path, "commit", "-m", commit_message)
            committed = True

    tracked_bad = [f for f in _git(path, "ls-files").splitlines() if _is_secret(f.split("/")[-1])]
    if tracked_bad:
        raise GitHubError(f"Refusing to push: tracked files look like secrets: {tracked_bad}. "
                          "Remove them from git (git rm --cached) and from history first.")

    refspec = f"refs/heads/{local_branch}:refs/heads/{remote_branch}"
    out = _git(path, "push", f"{GIT_BASE}/{repo}.git", refspec, env=_git_auth_env())
    return {"repo": repo, "pushed": f"{local_branch} -> {remote_branch}",
            "committed_first": committed, "git_output": out[-800:],
            "url": f"https://github.com/{repo}/tree/{remote_branch}"}


def preview_git_push(repo, local_path, branch="", remote_branch="", commit_message=""):
    path = str(Path(local_path).expanduser().resolve())
    current = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    lb = branch or current
    rb = remote_branch or lb
    changes = _git(path, "status", "--porcelain").splitlines()
    total = _git(path, "rev-list", "--count", lb, check=False)
    recent = _git(path, "log", "--oneline", "-n", "5", lb, check=False).splitlines()
    lines = [f"     git push: {path}", f"     {lb} -> {repo}:{rb}   ({total} commit(s) in history)"]
    lines += [f"       {c}" for c in recent]
    if commit_message:
        lines.append(f"     will first commit {len(changes)} uncommitted change(s): {commit_message!r}")
    elif changes:
        lines.append(f"     note: {len(changes)} uncommitted change(s) will NOT be pushed")
    lines.append("     (no force push: rejected if the remote branch has diverged)")
    return "\n".join(lines)


# ---- confirmation previews (shown before the user approves a write) ----
def preview_push_local_path(repo, branch, message, local_path, dest_dir="", create_from_branch=""):
    files, skipped = _collect_local_files(local_path, dest_dir)
    lines = [f"     repo: {repo}   branch: {branch}   message: {message!r}",
             f"     {len(files)} file(s) from {local_path}:"]
    lines += [f"       + {rp}" for rp, _ in files[:25]]
    if len(files) > 25:
        lines.append(f"       ... and {len(files) - 25} more")
    if skipped:
        lines.append("     will NOT be pushed:")
        lines += [f"       - {x}" for x in skipped[:10]]
    return "\n".join(lines)


def preview_push_files(repo, branch, message, files, create_from_branch=""):
    lines = [f"     repo: {repo}   branch: {branch}   message: {message!r}",
             f"     {len(files)} file(s):"]
    lines += [f"       + {f['path']} ({len(f['content'])} chars)" for f in files[:25]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool registry: schema + handler + whether it writes
# --------------------------------------------------------------------------- #
def _schema(props: dict, required: list[str]):
    return {"type": "object", "properties": props, "required": required}


REPO = {"type": "string", "description": "Repository in 'owner/name' form"}
STATE = {"type": "string", "enum": ["open", "closed", "all"]}
LIMIT = {"type": "integer", "description": "Max results"}

TOOLS = {
    "get_authenticated_user": (get_authenticated_user, False,
        "Get the GitHub profile of the account the token belongs to.", _schema({}, [])),
    "list_repos": (list_repos, False,
        "List repositories the account can access, most recently updated first.",
        _schema({"visibility": {"type": "string", "enum": ["all", "public", "private"]},
                 "limit": LIMIT}, [])),
    "get_repo": (get_repo, False, "Get details about one repository.",
        _schema({"repo": REPO}, ["repo"])),
    "list_issues": (list_issues, False, "List issues in a repository (excludes pull requests).",
        _schema({"repo": REPO, "state": STATE, "limit": LIMIT}, ["repo"])),
    "list_pull_requests": (list_pull_requests, False, "List pull requests in a repository.",
        _schema({"repo": REPO, "state": STATE, "limit": LIMIT}, ["repo"])),
    "list_commits": (list_commits, False, "List recent commits, optionally on a branch.",
        _schema({"repo": REPO, "branch": {"type": "string"}, "limit": LIMIT}, ["repo"])),
    "read_file": (read_file, False,
        "Read a file (or list a directory) from a repository at an optional ref.",
        _schema({"repo": REPO, "path": {"type": "string"}, "ref": {"type": "string"}},
                ["repo", "path"])),
    "search_code": (search_code, False,
        "Search code on GitHub using GitHub search syntax, e.g. 'TODO repo:owner/name'.",
        _schema({"query": {"type": "string"}, "limit": LIMIT}, ["query"])),
    "create_issue": (create_issue, True, "Create a new issue in a repository.",
        _schema({"repo": REPO, "title": {"type": "string"}, "body": {"type": "string"},
                 "labels": {"type": "array", "items": {"type": "string"}}}, ["repo", "title"])),
    "comment_on_issue": (comment_on_issue, True,
        "Add a comment to an issue or pull request.",
        _schema({"repo": REPO, "number": {"type": "integer"}, "body": {"type": "string"}},
                ["repo", "number", "body"])),
    "create_branch": (create_branch, True,
        "Create a new branch from an existing branch (defaults to the repo's default branch).",
        _schema({"repo": REPO, "branch": {"type": "string"}, "from_branch": {"type": "string"}},
                ["repo", "branch"])),
    "push_files": (push_files, True,
        "Commit one or more files with inline text content to a branch in ONE commit. "
        "Creates the branch if it does not exist. Use for code you generated.",
        _schema({"repo": REPO, "branch": {"type": "string"}, "message": {"type": "string"},
                 "files": {"type": "array", "items": {"type": "object",
                           "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                           "required": ["path", "content"]}},
                 "create_from_branch": {"type": "string",
                                        "description": "Base branch if the target branch is new"}},
                ["repo", "branch", "message", "files"])),
    "push_local_path": (push_local_path, True,
        "Push a local file or folder from the user's machine to a repo branch in ONE commit. "
        "Skips .git, node_modules, secrets (.env, keys) and large files. Creates the branch if needed.",
        _schema({"repo": REPO, "branch": {"type": "string"}, "message": {"type": "string"},
                 "local_path": {"type": "string", "description": "File or folder on this machine"},
                 "dest_dir": {"type": "string", "description": "Optional subfolder inside the repo"},
                 "create_from_branch": {"type": "string"}},
                ["repo", "branch", "message", "local_path"])),
    "create_pull_request": (create_pull_request, True,
        "Open a pull request from a head branch into a base branch (default: repo default).",
        _schema({"repo": REPO, "title": {"type": "string"}, "head": {"type": "string"},
                 "base": {"type": "string"}, "body": {"type": "string"},
                 "draft": {"type": "boolean"}}, ["repo", "title", "head"])),
    "git_push": (git_push, True,
        "Run a real `git push` of a LOCAL git repository on this machine to a GitHub repo, "
        "preserving commit history. Optionally commit pending changes first. Never force-pushes. "
        "Use for local repos; use push_local_path for plain folders without git history.",
        _schema({"repo": REPO, "local_path": {"type": "string", "description": "Path to the local git repo"},
                 "branch": {"type": "string", "description": "Local branch (default: checked-out branch)"},
                 "remote_branch": {"type": "string", "description": "Target branch name (default: same)"},
                 "commit_message": {"type": "string",
                                    "description": "If set, stage and commit all changes first"}},
                ["repo", "local_path"])),
    "create_repo": (create_repo, True,
        "Create a new repository under the authenticated user (private by default).",
        _schema({"name": {"type": "string"}, "private": {"type": "boolean"},
                 "description": {"type": "string"}}, ["name"])),
}

PREVIEWS = {"push_local_path": preview_push_local_path, "push_files": preview_push_files,
            "git_push": preview_git_push}

ANTHROPIC_TOOLS = [
    {"name": name, "description": desc, "input_schema": schema}
    for name, (_, _, desc, schema) in TOOLS.items()
]

SYSTEM = (
    "You are a GitHub assistant with access to the user's GitHub account through tools. "
    "Use the tools to answer questions with real data; never guess repository contents. "
    "Be concise. When pushing code, prefer a new feature branch and offer to open a pull request unless the user explicitly asks to push to the default branch; never push secrets. Use git_push for local git repositories (keeps history) and push_local_path for plain folders. Before performing a write action, make sure you have the details you need. "
    "If a tool returns an error, explain it plainly and suggest a fix (e.g. missing token permission)."
)


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
def run_tool(name: str, args: dict, auto_approve: bool) -> tuple[str, bool]:
    """Returns (result_text, is_error)."""
    if name not in TOOLS:
        return f"Unknown tool: {name}", True
    fn, writes, _, _ = TOOLS[name]

    if writes and not auto_approve:
        print(f"\n  ⚠️  Claude wants to run a WRITE action: {name}")
        try:
            if name in PREVIEWS:
                print(PREVIEWS[name](**args))
            else:
                print(f"     {json.dumps(args, indent=2)}")
        except (GitHubError, OSError, TypeError) as e:
            print(f"     (preview unavailable: {e})")
        if input("     Approve? [y/N] ").strip().lower() != "y":
            return "The user declined this action.", True

    try:
        out = json.dumps(fn(**args), indent=2, default=str)
    except GitHubError as e:
        return str(e), True
    except (TypeError, KeyError) as e:
        return f"Bad arguments: {e!r}", True
    except OSError as e:
        return f"Local file error: {e}", True
    except requests.RequestException as e:
        return f"Network error: {e}", True

    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + "\n...[truncated]"
    return out, False


def chat_turn(client: Anthropic, messages: list, auto_approve: bool):
    for _ in range(MAX_STEPS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                print(f"\n{block.text}")

        if resp.stop_reason != "tool_use":
            return

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"  → {block.name}({json.dumps(block.input)})")
                text, is_err = run_tool(block.name, block.input, auto_approve)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": text, "is_error": is_err})
        messages.append({"role": "user", "content": results})

    print("\n[Stopped: reached the maximum number of tool steps for one request.]")


def main():
    global gh, ALLOW_DEFAULT_BRANCH, GITHUB_TOKEN_VALUE
    parser = argparse.ArgumentParser(description="Claude-powered GitHub agent")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt (omit for interactive mode)")
    parser.add_argument("--yes", action="store_true", help="Auto-approve write actions")
    parser.add_argument("--allow-default-branch", action="store_true",
                        help="Allow pushing directly to a repo's default branch without the extra prompt")
    args = parser.parse_args()
    ALLOW_DEFAULT_BRANCH = args.allow_default_branch

    token = os.environ.get("GITHUB_TOKEN")
    if not token or not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set GITHUB_TOKEN and ANTHROPIC_API_KEY (see .env.example).")

    gh = GitHub(token)
    GITHUB_TOKEN_VALUE = token
    try:
        me = get_authenticated_user()
    except GitHubError as e:
        sys.exit(f"GitHub authentication failed: {e}")
    print(f"Connected to GitHub as @{me['login']}")

    client = Anthropic()
    messages: list = []

    if args.prompt:
        messages.append({"role": "user", "content": " ".join(args.prompt)})
        chat_turn(client, messages, args.yes)
        return

    print("Ask me about your repos, issues, PRs, code... (type 'exit' to quit)")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            messages.append({"role": "user", "content": q})
            chat_turn(client, messages, args.yes)


if __name__ == "__main__":
    main()
