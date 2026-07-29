"""Make the editable ``nqlab`` source importable in this virtualenv.

Why this exists
---------------
On some interpreters (observed with the ``python-build-standalone`` builds that
``uv`` provisions on this machine), the editable-install ``.pth`` file that
hatchling/uv writes is intermittently *not* applied by the ``site`` startup —
``site`` runs but the ``.pth`` path lines never reach ``sys.path``, so
``import nqlab`` fails even though the package is "installed". ``sitecustomize``
is imported through a different, reliable code path, so we use it to guarantee
``<repo>/src`` is importable.

This file is copied into the venv's ``site-packages`` by ``make install`` (it is
not part of the package). The source root is resolved relative to this file, so
it keeps working if the whole project (including its ``.venv``) is moved, as long
as the venv stays at ``<repo>/.venv``.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
# <repo>/.venv/lib/pythonX.Y/site-packages -> repo root is four levels up.
_repo_root = os.path.normpath(os.path.join(_here, "..", "..", "..", ".."))
_src = os.path.join(_repo_root, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)
