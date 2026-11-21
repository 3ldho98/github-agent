# GitHub Agent

Chat with your GitHub account (and push code to it) using Claude + a GitHub token.

## Setup
1. Create a **fine-grained token**: GitHub → Settings → Developer settings → Personal access tokens.
   - Repository access: only the repos you want
   - Permissions: Metadata (read), Contents (read & write), Issues (read & write), Pull requests (read & write)
   - Optional: Administration (write), only if you want the agent to create repos
2. `cp .env.example .env` and fill in `GITHUB_TOKEN` and `ANTHROPIC_API_KEY`
3. `pip install -r requirements.txt`
4. `python github_agent.py`  (or `python github_agent.py "your request"`)

## Tools
| Read (automatic) | Write (asks approval) |
|---|---|
| get_authenticated_user, list_repos, get_repo, list_issues, list_pull_requests, list_commits, read_file, search_code | create_issue, comment_on_issue, create_branch, push_files, push_local_path, git_push, create_pull_request, create_repo |

## Pushing code: which tool?
- **git_push**: a local git repo; real `git push`, history preserved, never force-pushes. Needs git >= 2.31.
- **push_local_path**: any folder; one new commit via the API (no git needed).
- **push_files**: code Claude writes during the chat; one commit via the API.

## Safety
- Every write shows a preview and needs a `y`. `--yes` skips this.
- Pushing to a repo's default branch requires typing the branch name (`--allow-default-branch` disables).
- Secrets (`.env`, keys, ...) are never pushed; `git_push` refuses if any are tracked.
  Note: it checks the current tree, not past history. Scan history yourself if a secret was ever committed.
- Your token is passed to git via environment config, never stored in `.git/config` or the remote URL.
- Never commit `.env`. Use short-lived tokens.
