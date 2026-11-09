from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_unitac_does_not_import_cuda_runtimes():
    repository = Path(__file__).resolve().parents[1]
    unisim_source = repository / "third_party" / "unisim" / "src"
    if not unisim_source.exists():
        unisim_source = repository.parent / "UniSim" / "src"
    environment = os.environ.copy()
    import_paths = [str(repository / "src")]
    if unisim_source.exists():
        import_paths.append(str(unisim_source))
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    code = """
import sys
import unitac
assert 'torch' not in sys.modules
assert 'cupy' not in sys.modules
assert callable(unitac.create_tactile_sensor)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=environment,
        check=True,
    )
