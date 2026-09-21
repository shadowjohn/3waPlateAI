"""No-clobber directory publication shared by generation and model export."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
from pathlib import Path


class PublicationError(Exception):
    """Base class for stable publication failures."""


class OutputExistsError(PublicationError, FileExistsError):
    """Raised when publication could modify an existing output target."""


def publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory without clobbering a raced target."""

    if output.exists():
        raise OutputExistsError(f"output already exists: {output}")

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PublicationError(
                "atomic no-clobber publication is unavailable on this Linux runtime"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise OutputExistsError(f"output already exists: {output}")
        if error_number in {errno.ENOSYS, errno.EINVAL}:
            raise PublicationError(
                "atomic no-clobber publication is unavailable on this filesystem"
            )
        raise OSError(error_number, os.strerror(error_number), output)

    if os.name == "nt":
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise OutputExistsError(f"output already exists: {output}") from exc
        return

    raise PublicationError(
        "atomic no-clobber publication is supported only on Linux and Windows"
    )


def remove_owned_staging(staging: Path, output: Path) -> None:
    """Remove only the sibling staging directory owned by ``output``."""

    if (
        staging.parent.resolve() == output.parent.resolve()
        and staging.name.startswith(f".{output.name}.partial-")
        and staging.is_dir()
    ):
        shutil.rmtree(staging)
