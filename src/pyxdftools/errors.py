"""Custom pyxdf-tools errors."""


class MetadataParseError (Exception):
    """Custom exception raised when failing to parse XDF metadata."""

    def __init__(self, stream_id):
        self.stream_id = stream_id

    def __str__(self):
        return f'Stream ID: {self.stream_id}'
