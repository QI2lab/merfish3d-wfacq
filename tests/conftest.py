import gc
import importlib.metadata as importlib_metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from hashlib import sha1
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / "pytest_tmp_env"
_RUNTIME_TMP = _WORKSPACE_TMP / "runtime_tmp"
_WORKSPACE_TMP.mkdir(exist_ok=True)
_RUNTIME_TMP.mkdir(exist_ok=True)
for _env_var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_env_var] = str(_RUNTIME_TMP)
tempfile.tempdir = str(_RUNTIME_TMP)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from pymmcore_gui.widgets._stage_explorer import _StageExplorer
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import _mmcore_plus
from pymmcore_widgets.mda._core_mda import MDAWidget

from merfish3d_wfacq.gui import MerfishFluidicsWidget
from merfish3d_wfacq.utils.data_io import read_fluidics_program

ANALYSIS_PR_NUMBER = 126
ANALYSIS_PR_SHA = "ac36ea8a0a2759c3e32343e9295382eb74997e57"

def _real_fluidics_cases() -> dict[str, dict[str, object]]:
    base = Path(__file__).parent
    return {
        "initial": {
            "name": "initial",
            "path": base / "initial.csv",
            "expected_fluidics_rounds": [1],
            "expected_imaging_rounds": [],
            "iterative_exp_order_text": "round,FITC,Rhodamine,Cy5\n1,0,1,2\n",
            "iterative_error": "RUN commands",
        },
        "full": {
            "name": "full",
            "path": base / "full.csv",
            "expected_fluidics_rounds": [1, 2, 3, 4, 5, 6, 7, 8],
            "expected_imaging_rounds": [1, 2, 3, 4, 5, 6, 7, 8],
            "iterative_exp_order_text": (
                "round,FITC,Rhodamine,Cy5\n"
                "1,0,1,2\n2,0,3,4\n3,0,5,6\n4,0,7,8\n"
                "5,0,9,10\n6,0,11,12\n7,0,13,14\n8,0,15,16\n"
            ),
            "iterative_error": None,
        },
    }


@pytest.fixture
def real_fluidics_case(request: pytest.FixtureRequest) -> dict[str, object]:
    case = dict(_real_fluidics_cases()[str(request.param)])
    case["program"] = read_fluidics_program(Path(case["path"]))
    case["exp_order_text"] = str(case["iterative_exp_order_text"])
    return case


def _process_qt_events() -> None:
    try:
        from pymmcore_gui._qt.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func: object, failed_path: str, _exc_info: object) -> None:
        try:
            Path(failed_path).chmod(0o700)
            func(failed_path)
        except Exception:
            pass

    for _attempt in range(40):
        _process_qt_events()
        gc.collect()
        try:
            shutil.rmtree(path, onerror=_onerror)
        except FileNotFoundError:
            return
        except Exception:
            pass
        if not path.exists():
            return
        time.sleep(0.25)


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in list(path.iterdir()):
        if child.is_dir():
            _remove_tree(child)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


@pytest.fixture(scope="session", autouse=True)
def workspace_temp_root() -> None:
    """Provide a sandbox-safe temporary root and remove it after the test session."""

    _cleanup_workspace_tmp()
    yield




@pytest.fixture
def workspace_tmp_path(request: pytest.FixtureRequest) -> Path:
    base = _WORKSPACE_TMP / "case_dirs"
    base.mkdir(parents=True, exist_ok=True)
    _clear_directory(base)
    node_hash = sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:10]
    candidate = base / f"t_{node_hash}_{os.getpid()}"
    suffix = 0
    while True:
        path = candidate.with_name(f"{candidate.name}_{suffix}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    yield path
    _remove_tree(path)

def _cleanup_workspace_tmp() -> None:
    _clear_directory(_WORKSPACE_TMP)
    _RUNTIME_TMP.mkdir(parents=True, exist_ok=True)


_cleanup_workspace_tmp()


def pytest_unconfigure(config: pytest.Config) -> None:
    del config
    _cleanup_workspace_tmp()


@pytest.fixture
def demo_core() -> CMMCorePlus:
    _mmcore_plus._instance = None
    core = CMMCorePlus()
    core.loadSystemConfiguration("MMConfig_demo.cfg")
    yield core
    try:
        core.waitForSystem()
    except Exception:
        pass
    try:
        core.unloadAllDevices()
    except Exception:
        pass
    _mmcore_plus._instance = None


@pytest.fixture
def offline_icons() -> None:
    svg_dir = _WORKSPACE_TMP / "icons"
    svg_dir.mkdir(exist_ok=True)
    counter = 0

    def mock_svg_path(*_key: str, color: str | None = None, **_kwargs: object) -> Path:
        nonlocal counter
        fill = color or "currentColor"
        svg_content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<rect width="24" height="24" fill="{fill}"/></svg>'
        )
        svg_file = svg_dir / f"icon_{counter}.svg"
        counter += 1
        svg_file.write_text(svg_content, encoding="utf-8")
        return svg_file

    with (
        patch("pymmcore_widgets.control._stage_widget.svg_path", mock_svg_path),
        patch("superqt.iconify.svg_path", mock_svg_path),
    ):
        yield


@pytest.fixture
def merfish_widget(
    qtbot: object,
    demo_core: CMMCorePlus,
    offline_icons: None,
) -> MerfishFluidicsWidget:
    mda_widget = MDAWidget(mmcore=demo_core)
    stage_explorer = _StageExplorer(mmcore=demo_core)
    widget = MerfishFluidicsWidget(
        mmcore=demo_core,
        mda_widget=mda_widget,
        stage_explorer=stage_explorer,
    )
    qtbot.addWidget(mda_widget)
    qtbot.addWidget(stage_explorer)
    qtbot.addWidget(widget)
    yield widget
    widget.close()
    stage_explorer.close()
    mda_widget.close()


def _require_qi2lab_datastore_api() -> type[object]:
    try:
        from merfish3danalysis.qi2labDataStore import qi2labDataStore
    except Exception as exc:
        sibling_repo = Path(__file__).resolve().parents[2] / "merfish3d-analysis"
        if sibling_repo.exists():
            try:
                head_sha = subprocess.run(
                    ["git", "-C", str(sibling_repo), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except Exception:
                head_sha = ""
            if head_sha == ANALYSIS_PR_SHA:
                sys.path.insert(0, str(sibling_repo / "src"))
                from merfish3danalysis.qi2labDataStore import qi2labDataStore

                return qi2labDataStore
        pytest.skip(
            "Install the analysis contract test extra pinned to "
            f"QI2lab/merfish3d-analysis PR #{ANALYSIS_PR_NUMBER} "
            f"({ANALYSIS_PR_SHA}) to run API validation: {exc}"
        )

    module_file = Path(inspect.getfile(qi2labDataStore))
    sibling_repo = next(
        (
            parent
            for parent in module_file.parents
            if parent.name == "merfish3d-analysis"
        ),
        None,
    )
    if sibling_repo is not None:
        head_sha = subprocess.run(
            ["git", "-C", str(sibling_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head_sha != ANALYSIS_PR_SHA:
            pytest.fail(
                "The local merfish3d-analysis checkout does not match the required "
                f"PR #{ANALYSIS_PR_NUMBER} ref. Expected {ANALYSIS_PR_SHA}, found "
                f"{head_sha}."
            )
        return qi2labDataStore

    try:
        distribution = importlib_metadata.distribution("merfish3d-analysis")
    except importlib_metadata.PackageNotFoundError:
        pytest.skip(
            "merfish3d-analysis is importable but not installed as a distribution that "
            "can be verified against PR #126."
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        pytest.skip(
            "The installed merfish3d-analysis distribution cannot be verified against "
            f"PR #{ANALYSIS_PR_NUMBER}; direct_url.json is missing."
        )

    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info", {})
    commit_id = vcs_info.get("commit_id")
    if commit_id != ANALYSIS_PR_SHA:
        pytest.fail(
            "The installed merfish3d-analysis distribution is not pinned to the "
            f"required PR #{ANALYSIS_PR_NUMBER} ref. Expected {ANALYSIS_PR_SHA}, "
            f"found {commit_id!r}."
        )
    return qi2labDataStore


@pytest.fixture
def qi2lab_datastore_cls() -> type[object]:
    return _require_qi2lab_datastore_api()










