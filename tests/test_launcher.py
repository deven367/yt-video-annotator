import subprocess
import sys
from pathlib import Path


def test_launcher_propagates_streamlit_failure_from_another_directory(tmp_path):
    launcher = Path(sys.executable).with_name("run_app")
    result = subprocess.run(
        [str(launcher), "--server.port=not-a-port"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not-a-port" in result.stderr
    assert "File does not exist" not in result.stderr
