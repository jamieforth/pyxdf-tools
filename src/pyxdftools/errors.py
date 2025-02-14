"""Custom pyxdf-tools errors."""


class XdfStreamLoadError(Exception):
    """Custom exception raised when failing to load XDF stream data."""

    def __init__(self, stream_id, cause):
        self.stream_id = stream_id
        self.__cause__ = cause

    def __str__(self):
        return f"Failed to load Stream {self.stream_id}: {self.__cause__}"


class XdfStreamParseError(Exception):
    """Custom exception raised when failing to parse XDF stream data."""

    def __init__(self, cause):
        self.__cause__ = cause

    def __str__(self):
        return f"Failed to parse stream: {self.__cause__}"


class XdfNotLoadedError(Exception):
    """Custom exception raised when XDF data has not yet been loaded."""

    def __str__(self):
        return "No streams loaded, call load() first."


class XdfAlreadyLoadedError(Exception):
    """Custom exception raised when attempting to re-load XDF data."""

    def __str__(self):
        return "Streams already loaded."


class NoLoadableStreamsError(Exception):
    """Custom exception raised when no loadable XDF stream data exist."""

    def __init__(self, select_streams):
        self.select_streams = select_streams

    def __str__(self):
        return f"No loadable Streams matching {self.select_streams}"
