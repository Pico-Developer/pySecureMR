"""Utilities for defining custom operators executed inside the native pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Dict

from ..bindings import BindingsUnavailableError, bindings_available, load_bindings

try:  # pragma: no cover - depends on bindings availability
    if bindings_available():
        load_bindings()
        from ..bindings.linux import _securemr  # type: ignore
    else:
        _securemr = None
except Exception:  # noqa: BLE001
    _securemr = None


_TOKEN_TO_IMPLEMENTATION: Dict[str, "CustomOperatorBase"] = {}


class CustomOperatorHandle:
    """Lightweight wrapper around a registered custom operator token."""

    def __init__(self, implementation: "CustomOperatorBase") -> None:
        self._implementation = implementation
        if _securemr is None:
            raise BindingsUnavailableError(
                "Custom operators require native bindings to be loaded."
            )
        self._token: Optional[str] = _securemr.register_custom_operator(implementation)
        self._released = False
        if self._token is not None:
            _TOKEN_TO_IMPLEMENTATION[self._token] = implementation

    @property
    def token(self) -> str:
        if self._token is None:
            raise RuntimeError("Custom operator handle has been released")
        return self._token

    def configs(self) -> List[str]:
        return [f"token:{self.token}"]

    def release(self) -> None:
        if self._token is None or self._released:
            return
        if _securemr is None:
            raise BindingsUnavailableError(
                "Custom operators require native bindings to be loaded."
            )
        _securemr.release_custom_operator(self._token)
        self._released = True
        if self._token in _TOKEN_TO_IMPLEMENTATION:
            _TOKEN_TO_IMPLEMENTATION.pop(self._token, None)
        self._token = None

    def __enter__(self) -> "CustomOperatorHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.release()
        except Exception:
            pass


class CustomOperatorBase(ABC):
    """Base class for python-defined operators executed via :class:`Operator_Custom`."""

    verify_operand = None
    verify_result = None

    def __init__(
        self,
        operand_names: Sequence[str],
        result_names: Sequence[str],
    ) -> None:
        if not operand_names:
            raise ValueError("operand_names must not be empty")
        if not result_names:
            raise ValueError("result_names must not be empty")

        self.operand_names = list(operand_names)
        self.result_names = list(result_names)
        self._handle: Optional[CustomOperatorHandle] = None

    @abstractmethod
    def compute(self, task_id: int, operands, results) -> None:
        """Perform computation for the current task.

        Args:
            task_id: Task identifier provided by the engine.
            operands: Mutable list of tensors representing the operator inputs.
            results: Mutable list of tensors representing the operator outputs.
        """

    def register(self) -> CustomOperatorHandle:
        """Register the custom operator with the native registry."""

        if self._handle is None:
            self._handle = CustomOperatorHandle(self)
        return self._handle

    def release(self) -> None:
        if self._handle is not None:
            self._handle.release()
            self._handle = None


def create_operator_configs(operator: CustomOperatorBase) -> List[str]:
    """Register ``operator`` and return the configuration list for pipeline allocation."""

    handle = operator.register()
    return handle.configs()


def get_registered_custom_operator(token: str) -> Optional[CustomOperatorBase]:
    """Return the Python implementation previously registered for ``token``."""

    return _TOKEN_TO_IMPLEMENTATION.get(token)
