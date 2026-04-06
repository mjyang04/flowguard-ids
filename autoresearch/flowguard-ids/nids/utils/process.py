from __future__ import annotations

import subprocess


def run_command(cmd: list[str], logger) -> None:
    """Run a subprocess command and raise RuntimeError on non-zero exit."""
    logger.info("Running: %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")
