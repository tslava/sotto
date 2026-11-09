"""sotto — телеграм-бот-персонаж для одного чата."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sotto")
except PackageNotFoundError:
    __version__ = "0.0.0"
