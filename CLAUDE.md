# Project: kszx

  - Unless otherwise specified, use units (comoving Mpc) for distances and times, and (Msol) for masses.
    Velocities should be dimensioness (c=1), not km/s.
    
    Note that we don't use h-units. If you're calling an external library which does use h-units, make
    sure to insert factors of h (or h^{-1}) if needed.
    
  - Put all plans in plans/*.md, and don't add them to git.
    Assume plans are ephemereal -- don't reference them in documentation (including docstrings).

  - Don't add memories to MEMORY.md. Instead, if there is something non-obvious about the code that we should
    remember in the future, add comments to the code (or edit documentation / docstrings) as appropriate.

  - Only do 'git commit' if the most recent prompt specifically asks for it.

  - If I ask you to edit an .ipynb notebook, and the prompt does not contain "I've saved", then
    please ask whether the file on disk is up-to-date. (This is to avoid a situation where I ask
    you to edit a notebook with my unsaved edits in memory. I'll usually remember to save the
    notebook to disk before prompting you to edit it, but sometimes I'll forget.)
    
  - Environment setup: every Bash command must run inside both the `kszx` conda env
    and the project venv. The simplest way is to prefix commands with:
        conda activate kszx && source .venv/bin/activate &&
    For example:
        conda activate kszx && source .venv/bin/activate && python my_script.py
    Do NOT use `conda run -n kszx` (it has quoting issues and doesn't activate the venv).  

  - Useful documentation is in `docs/source/*.rst` (in addition to docstrings).
