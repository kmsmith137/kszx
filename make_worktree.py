#!/usr/bin/env python
#
# I'm experimenting with an ad hoc system for creating named worktrees (mainly for LLM agents),
# consisting of two files: .envrc and make_worktree.py.
#
# The envrc is just a convenience that automatically switches between envs (conda envs + venvs)
# when switching between worktrees. This is awkward to set up -- here are the instructions.
#
# Usage: 'python make_worktree.py DIRNAME' will:
#   - create a worktree at DIRNAME
#   - create a venv in the worktree
#   - install .envrc and do 'direnv allow'

import os
import sys
import shutil
import subprocess


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dirname>", file=sys.stderr)
        sys.exit(1)

    # Get toplevel directory.
    toplevel = os.getcwd()

    # Fail if run from a git worktree (as opposed to the main repo).
    git_common_dir = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()
    git_dir = subprocess.check_output(['git', 'rev-parse', '--git-dir'], text=True).strip()
    if os.path.realpath(git_common_dir) != os.path.realpath(git_dir):
        print("Error: this script must be run from the main git repo, not a worktree.", file=sys.stderr)
        sys.exit(1)

    worktree_path = os.path.realpath(sys.argv[1])
    branch_name = os.path.basename(worktree_path)

    if os.path.exists(worktree_path):
        print(f"Error: {worktree_path} already exists.", file=sys.stderr)
        sys.exit(1)

    # Check if branch already exists.
    ret = subprocess.run(['git', 'rev-parse', '--verify', branch_name], capture_output=True)
    if ret.returncode == 0:
        print(f"Error: branch '{branch_name}' already exists. (To delete, do 'git branch -d {branch_name}'", file=sys.stderr)
        sys.exit(1)

    # Create git worktree.
    print(f"Creating git worktree: {worktree_path} (branch '{branch_name}' from HEAD)")
    subprocess.check_call(['git', 'worktree', 'add', '-b', branch_name, worktree_path, 'HEAD'])

    # Create venv.
    print(f"Creating venv: {worktree_path}/.venv")
    subprocess.check_call([sys.executable, '-m', 'venv', '--system-site-packages', os.path.join(worktree_path, '.venv')])

    # Copy dot_envrc -> .envrc.
    dot_envrc_src = os.path.join(toplevel, 'dot_envrc')
    dot_envrc_dst = os.path.join(worktree_path, '.envrc')
    print(f"Copying dot_envrc -> {dot_envrc_dst}")
    shutil.copy2(dot_envrc_src, dot_envrc_dst)

    # Run direnv allow.
    print(f"Running 'direnv allow' in {worktree_path}")
    subprocess.check_call(['direnv', 'allow'], cwd=worktree_path)

    print()
    print("Done! To finish setup:")
    print(f"  cd {worktree_path}")
    print(f"  source .venv/bin/activate  # automatic if using direnv")
    print(f"  pip install --no-build-isolation -v -e .")
    print()
    print("Optional (only needed if running Jupyter notebooks):")
    print("  pip install ipykernel   # if not already installed")
    print(f"  python -m ipykernel install --user --name {branch_name} --display-name \"{branch_name}\"")


if __name__ == '__main__':
    main()
