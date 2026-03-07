# Project: kszx

  - Put all plans in plans/*.md, and don't add them to git.
    Assume plans are ephemereal -- don't reference them in documentation (including docstrings).

  - Don't add memories to MEMORY.md. Instead, if there is something non-obvious about the code that we should
    remember in the future, add comments to the code (or edit documentation / docstrings) as appropriate.

  - Only do 'git commit' if the most recent prompt specifically asks for it.
  
  - All shell commands must be run in the `kszx` conda environment. Prefix every Bash command with `conda run -n kszx --no-capture-output` or activate the environment first.

  - Useful documentation is in `docs/source/*.rst` (in addition to docstrings).
