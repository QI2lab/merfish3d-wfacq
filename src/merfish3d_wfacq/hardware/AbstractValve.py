"""
Abstract class for controlling a hardware valve. Function names were derived from original valveChain and hamilton classes.

George Emanuel
2/16/2018
"""

from abc import ABC, abstractmethod


class AbstractValve(ABC):
    @abstractmethod
    def changePort(self, valve_ID, port_ID, direction=0):
        """Move one valve to the requested port.

        Parameters
        ----------
        valve_ID : Any
            Valve identifier.
        port_ID : Any
            Target port identifier.
        direction : Any, optional
            Requested rotation direction.
        """

        pass

    @abstractmethod
    def howManyValves(self):
        """Return the number of valves in the chain.

        Returns
        -------
        Any
            Number of valves in the chain.
        """

        pass

    @abstractmethod
    def close(self):
        """Close the valve controller."""

        pass

    @abstractmethod
    def getDefaultPortNames(self, valve_ID):
        """Return default labels for the requested valve ports.

        Parameters
        ----------
        valve_ID : Any
            Valve identifier.

        Returns
        -------
        Any
            Default port labels.
        """

        pass

    @abstractmethod
    def howIsValveConfigured(self, valve_ID):
        """Return the configuration string for one valve.

        Parameters
        ----------
        valve_ID : Any
            Valve identifier.

        Returns
        -------
        Any
            Valve configuration string.
        """

        pass

    @abstractmethod
    def getStatus(self, valve_ID):
        """Return movement and location status for one valve.

        Parameters
        ----------
        valve_ID : Any
            Valve identifier.

        Returns
        -------
        Any
            Valve status payload.
        """

        pass

    @abstractmethod
    def resetChain(self):
        """Reinitialize the valve chain."""

        pass

    @abstractmethod
    def getRotationDirections(self, valve_ID):
        """Return human-readable rotation directions for one valve.

        Parameters
        ----------
        valve_ID : Any
            Valve identifier.

        Returns
        -------
        Any
            Rotation direction labels.
        """

        pass
