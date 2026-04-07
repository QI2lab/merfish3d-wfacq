import numpy as np


class ZDriftEstimator:
    """Estimate integer-plane z drift between two stacks."""

    def estimate_from_reference_plane(
        self,
        reference_plane: np.ndarray,
        moving_stack: np.ndarray,
        *,
        reference_z_um: float,
        moving_z_positions_um: list[float],
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        """Estimate drift from a single reference plane and swept stack.

        Parameters
        ----------
        reference_plane : np.ndarray
            Reference fiducial image with shape ``(y, x)``.
        moving_stack : np.ndarray
            Newly acquired drift stack with shape ``(z, y, x)``.
        reference_z_um : float
            Nominal z position of the stored reference plane.
        moving_z_positions_um : list[float]
            Physical z positions corresponding to ``moving_stack``.
        current_offset_um : float, optional
            Previously applied z offset.

        Returns
        -------
        dict[str, float | int]
            Estimated plane index, incremental shift, and absolute offset.
        """

        reference = self._normalize_plane(reference_plane)
        moving = self._normalize_stack(moving_stack)
        if len(moving_z_positions_um) != moving.shape[0]:
            raise ValueError(
                "Moving z positions must match the acquired drift-stack depth."
            )

        best_index = 0
        best_score = float("-inf")
        for index, plane in enumerate(moving):
            numerator = float(np.sum(reference * plane))
            denominator = float(
                np.linalg.norm(reference.ravel()) * np.linalg.norm(plane.ravel())
            )
            score = numerator / denominator if denominator else float("-inf")
            if score > best_score:
                best_score = score
                best_index = index

        absolute_offset_um = float(moving_z_positions_um[best_index] - reference_z_um)
        shift_um = float(absolute_offset_um - current_offset_um)
        return {
            "shift_planes": int(best_index),
            "shift_um": shift_um,
            "absolute_offset_um": absolute_offset_um,
        }

    def estimate_plane_shift(
        self, reference_stack: np.ndarray, moving_stack: np.ndarray
    ) -> int:
        """Estimate the best integer-plane shift between two stacks.

        Parameters
        ----------
        reference_stack : np.ndarray
            Reference stack with shape ``(z, y, x)``.
        moving_stack : np.ndarray
            Moving stack with shape ``(z, y, x)``.

        Returns
        -------
        int
            Best-matching integer z-plane shift.
        """

        reference = self._normalize_stack(reference_stack)
        moving = self._normalize_stack(moving_stack)
        max_shift = min(reference.shape[0], moving.shape[0]) - 1
        best_shift = 0
        best_score = float("-inf")

        for shift in range(-max_shift, max_shift + 1):
            ref_slice, mov_slice = self._overlap(reference, moving, shift)
            if ref_slice.size == 0 or mov_slice.size == 0:
                continue
            numerator = float(np.sum(ref_slice * mov_slice))
            denominator = float(
                np.linalg.norm(ref_slice.ravel()) * np.linalg.norm(mov_slice.ravel())
            )
            score = numerator / denominator if denominator else float("-inf")
            if score > best_score:
                best_score = score
                best_shift = shift

        return int(best_shift)

    def estimate(
        self,
        reference_stack: np.ndarray,
        moving_stack: np.ndarray,
        *,
        z_step_um: float,
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        """Estimate drift between two stacks sampled on the same z grid.

        Parameters
        ----------
        reference_stack : np.ndarray
            Reference stack with shape ``(z, y, x)``.
        moving_stack : np.ndarray
            Moving stack with shape ``(z, y, x)``.
        z_step_um : float
            Z spacing between adjacent planes in microns.
        current_offset_um : float, optional
            Previously applied z offset.

        Returns
        -------
        dict[str, float | int]
            Estimated plane shift, incremental shift, and absolute offset.
        """

        shift_planes = self.estimate_plane_shift(reference_stack, moving_stack)
        shift_um = shift_planes * float(z_step_um)
        return {
            "shift_planes": int(shift_planes),
            "shift_um": float(shift_um),
            "absolute_offset_um": float(current_offset_um + shift_um),
        }

    @staticmethod
    def _normalize_stack(stack: np.ndarray) -> np.ndarray:
        """Normalize each plane in a stack to zero mean and unit variance.

        Parameters
        ----------
        stack : np.ndarray
            Stack with shape ``(z, y, x)``.

        Returns
        -------
        np.ndarray
            Normalized stack.
        """

        array = np.asarray(stack, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("ZDriftEstimator expects stacks with shape (z, y, x).")
        array = array - array.mean(axis=(1, 2), keepdims=True)
        std = array.std(axis=(1, 2), keepdims=True)
        std[std == 0] = 1.0
        return array / std

    @staticmethod
    def _normalize_plane(image: np.ndarray) -> np.ndarray:
        """Normalize one plane to zero mean and unit variance.

        Parameters
        ----------
        image : np.ndarray
            Image with shape ``(y, x)``.

        Returns
        -------
        np.ndarray
            Normalized image.
        """

        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("ZDriftEstimator expects reference planes with shape (y, x).")
        array = array - array.mean()
        std = float(array.std())
        if std == 0:
            std = 1.0
        return array / std

    @staticmethod
    def _overlap(
        reference: np.ndarray, moving: np.ndarray, shift: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the overlapping stack regions for a candidate z shift.

        Parameters
        ----------
        reference : np.ndarray
            Reference stack.
        moving : np.ndarray
            Moving stack.
        shift : int
            Candidate plane shift.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Overlapping stack views for the candidate shift.
        """

        if shift >= 0:
            if shift >= moving.shape[0]:
                return reference[:0], moving[:0]
            length = min(reference.shape[0], moving.shape[0] - shift)
            return reference[:length], moving[shift : shift + length]

        offset = abs(shift)
        if offset >= reference.shape[0]:
            return reference[:0], moving[:0]
        length = min(reference.shape[0] - offset, moving.shape[0])
        return reference[offset : offset + length], moving[:length]
