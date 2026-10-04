from __future__ import annotations

from abc import ABC, abstractmethod


class RequestCapacityError(RuntimeError):
    pass


class RequestGate(ABC):
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def acquire(self, tenant_id: str, user_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def renew(self, tenant_id: str, user_id: str, token: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def release(self, tenant_id: str, user_id: str, token: str) -> None:
        raise NotImplementedError
