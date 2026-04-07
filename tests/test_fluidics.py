import pytest

from merfish3d_wfacq.fluidics import FluidicsController


class _FakePump:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | str]] = []

    def startFlow(self, rate: float, direction: str = "Forward") -> None:
        self.calls.append(("start", rate))
        self.calls.append(("direction", direction))

    def stopFlow(self) -> None:
        self.calls.append(("stop", 0.0))


class _FakeValves:
    def __init__(self) -> None:
        self.ports: list[tuple[int, int]] = []

    def changePort(self, *, valve_ID: int, port_ID: int) -> None:
        self.ports.append((valve_ID, port_ID))


def test_refresh_requires_operator_confirmation_handler() -> None:
    controller = FluidicsController(
        {"settle_time_s": 0.0},
        [{"round": 1, "source": "REFRESH", "time": 0.0, "pump": 0.0}],
    )
    controller._pump = _FakePump()
    controller._valves = _FakeValves()

    with pytest.raises(RuntimeError, match="no operator confirmation handler"):
        controller.execute_round(1)
