from __future__ import annotations

import os
import socket
import ssl


def main():
    hostname = os.environ["HEALTHCHECK_HOSTNAME"]
    context = ssl.create_default_context(cafile="/run/secrets/tls_certificate")
    with socket.create_connection(("127.0.0.1", 8443), timeout=5) as connection:
        with context.wrap_socket(connection, server_hostname=hostname) as secure:
            secure.sendall(
                f"GET /readyz HTTP/1.1\r\nHost: {hostname}\r\nConnection: close\r\n\r\n".encode(
                    "ascii"
                )
            )
            response = secure.recv(4096)
            if not response.startswith(b"HTTP/1.1 200 "):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
