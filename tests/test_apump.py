from merfish3d_wfacq.hardware.APump import APump


def test_simulated_apump_behaves_like_simulated_backend() -> None:
    pump = APump({"simulate_pump": True, "verbose": False})
    assert pump.identification == "SIMULATED APUMP"

    pump.startFlow(12.5, direction="Reverse")
    status = pump.getStatus()

    assert status[0] == "Flowing"
    assert status[2] == "Reverse"
    assert any(entry.startswith("BUFFERED") for entry in pump.command_log)

    pump.stopFlow()
    stopped = pump.getStatus()
    assert stopped[0] == "Stopped"
