"""Console entry points declared in ``pyproject.toml``.

Each function is a thin shim that imports the corresponding script module
and forwards ``sys.argv`` to its argparse-driven ``main()`` function.  This
indirection keeps the scripts directly runnable as ``python -m scripts.train``
while also exposing ``r3dc-train`` / ``r3dc-eval`` / ``r3dc-infer`` commands.
"""

from __future__ import annotations

import sys


def train_main() -> None:
    from scripts import train as _train  # type: ignore

    sys.exit(_train.main())


def eval_main() -> None:
    from scripts import eval as _eval  # type: ignore

    sys.exit(_eval.main())


def infer_main() -> None:
    from scripts import infer as _infer  # type: ignore

    sys.exit(_infer.main())
