"""CI-only Python network/import guard, activated through PYTHONPATH.

Dependency installation runs before this directory is put on PYTHONPATH. Child
Python processes inherit the guard. This is a regression check for accidental
Python networking, not an operating-system network sandbox.
"""

import importlib.abc
import os
import socket
import sys


def deny_network(*args, **kwargs):
    raise RuntimeError("Offline CI attempted Python network access.")


for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
             "gethostbyaddr"):
    setattr(socket, name, deny_network)
for name in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, deny_network)
socket._mrl_offline_guard = True


class NoHardwareImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cv2", "mediapipe", "torch", "sounddevice", "pyaudio"}:
            raise RuntimeError("Minimal offline CI imported optional hardware stack: " + fullname)


if os.environ.get("MRL_OFFLINE_MINIMAL") == "1":
    sys.meta_path.insert(0, NoHardwareImports())
