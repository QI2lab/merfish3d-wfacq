#!/usr/bin/python
"""Peristaltic pump control with real and simulated backends."""

from abc import ABC, abstractmethod
from typing import Any

acknowledge = "\x06"
start = "\x0a"
stop = "\x0d"


class _PumpBackend(ABC):
    """Low-level transport/backend abstraction for the pump."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    @abstractmethod
    def send_immediate(self, unit_number: int, command: str) -> str:
        """Send a command that returns an immediate response."""

    @abstractmethod
    def send_buffered(self, unit_number: int, command: str) -> None:
        """Send a buffered command."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the currently selected unit."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


class _SerialPumpBackend(_PumpBackend):
    """Real serial-port backend."""

    def __init__(self, state: dict[str, Any], *, com_port: str) -> None:
        """Open the serial connection for a physical pump.

        Parameters
        ----------
        state : dict[str, Any]
            Shared pump state dictionary.
        com_port : str
            Serial port for the physical pump.
        """

        super().__init__(state)
        import serial

        self._serial = serial.Serial(
            port=com_port,
            baudrate=19200,
            parity=serial.PARITY_EVEN,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_TWO,
            timeout=0.1,
        )

    def send_immediate(self, unit_number: int, command: str) -> str:
        """Send an immediate-response command to the selected unit.

        Parameters
        ----------
        unit_number : int
            Pump unit address.
        command : str
            Immediate-response command string.

        Returns
        -------
        str
            Pump response string.
        """

        self._select_unit(unit_number)
        self._send_string(command[0])
        new_character = self._get_response()
        response = ""
        while new_character and not (ord(new_character) & 0x80):
            response += new_character.decode()
            self._send_string(acknowledge)
            new_character = self._get_response()

        if new_character:
            response += chr(ord(new_character) & ~0x80)

        self.disconnect()
        return response

    def send_buffered(self, unit_number: int, command: str) -> None:
        """Send a buffered command to the selected unit.

        Parameters
        ----------
        unit_number : int
            Pump unit address.
        command : str
            Buffered command string.
        """

        self._select_unit(unit_number)
        self._send_and_acknowledge(start + command + stop)
        self.disconnect()

    def disconnect(self) -> None:
        """Disconnect from the currently selected unit."""

        self._send_and_acknowledge("\xff")

    def close(self) -> None:
        """Close the serial connection."""

        self._serial.close()

    def _select_unit(self, unit_number: int) -> bool:
        """Select one pump unit on the serial bus.

        Parameters
        ----------
        unit_number : int
            Pump unit address.

        Returns
        -------
        bool
            ``True`` when the unit acknowledges selection.
        """

        dev_select = chr(0x80 | unit_number)
        self._send_string(dev_select)
        return self._get_response() == dev_select.encode()

    def _send_and_acknowledge(self, string: str) -> None:
        """Send a command string and wait for per-character acknowledgements.

        Parameters
        ----------
        string : str
            Raw command string to send.
        """

        for char in string:
            self._send_string(char)
            self._get_response()

    def _send_string(self, string: str) -> None:
        """Send raw bytes to the serial backend.

        Parameters
        ----------
        string : str
            Raw string to send.
        """

        self._serial.write(string.encode())

    def _get_response(self) -> bytes:
        """Read one response byte from the serial backend.

        Returns
        -------
        bytes
            One byte read from the serial backend.
        """

        return self._serial.read()


class _SimulatedPumpBackend(_PumpBackend):
    """Deterministic in-memory backend used for tests and demos."""

    def send_immediate(self, unit_number: int, command: str) -> str:
        """Return simulated immediate-response pump data.

        Parameters
        ----------
        unit_number : int
            Pump unit address.
        command : str
            Immediate-response command string.

        Returns
        -------
        str
            Simulated response string.
        """

        self.state["command_log"].append(f"IMMEDIATE:{unit_number}:{command}")
        if command == "%":
            return "SIMULATED APUMP"
        if command == "R":
            if float(self.state["speed"]) > 0:
                sign = "+" if self.state["direction"] == "Forward" else "-"
            else:
                sign = " "
            control = "R" if self.state["remote_enabled"] else "K"
            return f"{sign}{float(self.state['speed']):.2f}{control}"
        return ""

    def send_buffered(self, unit_number: int, command: str) -> None:
        """Record a simulated buffered command.

        Parameters
        ----------
        unit_number : int
            Pump unit address.
        command : str
            Buffered command string.
        """

        self.state["command_log"].append(f"BUFFERED:{unit_number}:{command}")

    def disconnect(self) -> None:
        """Record a simulated disconnect event."""

        self.state["command_log"].append("DISCONNECT")

    def close(self) -> None:
        """Record closure of the simulated backend."""

        self.state["command_log"].append("CLOSE")


class APump:
    """Facade for the Gibson peristaltic pump."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        """Create a real or simulated pump facade.

        Parameters
        ----------
        parameters : dict[str, Any] or None, optional
            Pump configuration dictionary.
        """

        parameters = parameters or {}

        self.com_port = parameters.get("pump_com_port", "COM3")
        self.pump_ID = int(parameters.get("pump_ID", 30))
        self.verbose = bool(parameters.get("verbose", True))
        self.simulate = bool(parameters.get("simulate_pump", True))
        self.serial_verbose = bool(parameters.get("serial_verbose", False))
        self.flip_flow_direction = bool(parameters.get("flip_flow_direction", False))

        self._state: dict[str, Any] = {
            "speed": 0.0,
            "direction": "Forward",
            "remote_enabled": False,
            "flow_status": "Stopped",
            "command_log": [],
        }
        if self.simulate:
            self._backend: _PumpBackend = _SimulatedPumpBackend(self._state)
        else:
            self._backend = _SerialPumpBackend(self._state, com_port=self.com_port)

        self.flow_status = "Stopped"
        self.speed = 0.0
        self.direction = "Forward"

        self.disconnect()
        self.enableRemoteControl(True)
        self.startFlow(self.speed, self.direction)
        self.identification = self.getIdentification()

    @property
    def command_log(self) -> list[str]:
        """Return the backend command log.

        Returns
        -------
        list[str]
            Recorded backend command log.
        """
        return list(self._state["command_log"])

    def getIdentification(self) -> str:
        """Return the pump identification string.

        Returns
        -------
        str
            Pump identification string.
        """

        return self.sendImmediate(self.pump_ID, "%")

    def enableRemoteControl(self, remote: bool) -> None:
        """Enable or disable remote-control mode.

        Parameters
        ----------
        remote : bool
            Whether remote control should be enabled.
        """

        if remote:
            self.sendBuffered(self.pump_ID, "SR")
        else:
            self.sendBuffered(self.pump_ID, "SK")
        self._state["remote_enabled"] = bool(remote)

    def readDisplay(self) -> str:
        """Read the pump front-panel display string.

        Returns
        -------
        str
            Raw display string returned by the pump.
        """

        return self.sendImmediate(self.pump_ID, "R")

    def getStatus(self) -> tuple[str, float, str, str, str, str]:
        """Return decoded pump status information.

        Returns
        -------
        tuple[str, float, str, str, str, str]
            Decoded status tuple.
        """

        message = self.readDisplay()
        if not message:
            direction = str(self._state["direction"])
            speed = float(self._state["speed"])
            control = "Remote" if self._state["remote_enabled"] else "Keypad"
            status = "Flowing" if speed > 0 else "Stopped"
            return (status, speed, direction, control, "Disabled", "No Error")

        if self.flip_flow_direction:
            direction = {" ": "Not Running", "-": "Forward", "+": "Reverse"}.get(
                message[0], "Unknown"
            )
        else:
            direction = {" ": "Not Running", "+": "Forward", "-": "Reverse"}.get(
                message[0], "Unknown"
            )

        status = "Stopped" if direction == "Not Running" else "Flowing"
        control = {"K": "Keypad", "R": "Remote"}.get(message[-1], "Unknown")

        try:
            speed = float(message[1 : len(message) - 1])
        except ValueError:
            speed = float(self._state["speed"])

        return (status, speed, direction, control, "Disabled", "No Error")

    def close(self) -> None:
        """Disable remote mode and close the backend."""

        self.enableRemoteControl(False)
        self._backend.close()

    def setFlowDirection(self, forward: bool) -> None:
        """Set the pump flow direction.

        Parameters
        ----------
        forward : bool
            Whether the flow direction should be forward.
        """

        if self.flip_flow_direction:
            command = "K<" if forward else "K>"
        else:
            command = "K>" if forward else "K<"
        self.sendBuffered(self.pump_ID, command)
        self._state["direction"] = "Forward" if forward else "Reverse"

    def setSpeed(self, rotation_speed: float) -> None:
        """Set the pump speed in rotations per minute.

        Parameters
        ----------
        rotation_speed : float
            Requested pump speed.
        """

        rotation_speed = float(rotation_speed)
        if 0 <= rotation_speed <= 48:
            rotation_int = int(rotation_speed * 100)
            self.sendBuffered(self.pump_ID, f"R{rotation_int:04d}")
            self._state["speed"] = rotation_speed
            self._state["flow_status"] = (
                "Flowing" if rotation_speed > 0 else "Stopped"
            )

    def startFlow(self, speed: float, direction: str = "Forward") -> None:
        """Start pump flow at the requested speed and direction.

        Parameters
        ----------
        speed : float
            Requested pump speed.
        direction : str, optional
            Requested flow direction.
        """

        self.setSpeed(speed)
        self.setFlowDirection(direction == "Forward")
        self.flow_status = str(self._state["flow_status"])
        self.speed = float(self._state["speed"])
        self.direction = str(self._state["direction"])

    def stopFlow(self) -> bool:
        """Stop pump flow.

        Returns
        -------
        bool
            ``True`` when the stop command completes.
        """

        self.setSpeed(0.0)
        self.flow_status = str(self._state["flow_status"])
        self.speed = float(self._state["speed"])
        return True

    def sendImmediate(self, unitNumber: int, command: str) -> str:
        """Send an immediate-response command through the backend.

        Parameters
        ----------
        unitNumber : int
            Pump unit address.
        command : str
            Immediate-response command string.

        Returns
        -------
        str
            Pump response string.
        """

        return self._backend.send_immediate(unitNumber, command)

    def sendBuffered(self, unitNumber: int, command: str) -> None:
        """Send a buffered command through the backend.

        Parameters
        ----------
        unitNumber : int
            Pump unit address.
        command : str
            Buffered command string.
        """

        self._backend.send_buffered(unitNumber, command)

    def disconnect(self) -> None:
        """Disconnect the current backend unit selection."""

        self._backend.disconnect()
