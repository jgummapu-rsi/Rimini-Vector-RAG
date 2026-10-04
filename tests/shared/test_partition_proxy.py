import socket
from urllib.parse import urlsplit

import pytest

from eval.network_load import PartitionProxy


def test_real_tcp_partition_and_recovery(storage_settings):
    address = urlsplit(storage_settings.redis_url)
    proxy = PartitionProxy(address.hostname, address.port or 6379)
    try:
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=1) as connection:
            connection.settimeout(0.1)
            connection.sendall(b"*1\r\n$4\r\nPING\r\n")
            assert connection.recv(128) == b"+PONG\r\n"
            proxy.partitioned.set()
            connection.sendall(b"*1\r\n$4\r\nPING\r\n")
            with pytest.raises(TimeoutError):
                connection.recv(128)
            proxy.partitioned.clear()
            connection.settimeout(1)
            assert connection.recv(128) == b"+PONG\r\n"
    finally:
        proxy.close()
    assert not proxy.thread.is_alive()
