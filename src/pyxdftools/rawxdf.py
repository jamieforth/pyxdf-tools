"""Thin wrapper for raw XDF data."""

import numpy as np
import pyxdf

from .errors import DataStreamLoadError


class RawXdf:
    """Thin wrapper for raw XDF data.

    Provides convenience methods for accessing the raw XDF
    list-of-dictionaries data structure.

    Attributes:
        filename: XDF file - string or Path.
        verbose: Boolean determining additional logging.
    """

    filename = None
    verbose = None

    # Boolean indicating if file has been loaded.
    _loaded = False

    # Dictionary containing file header information.
    _header = None

    # List of dictionaries corresponding to each loaded stream.
    _streams = None

    def __init__(self, filename, verbose=False):
        """Initialise XDF file."""
        self.filename = filename
        self.verbose = verbose

    def resolve_streams(self):
        """Resolve streams in the current file."""
        return pyxdf.resolve_streams(str(self.filename))

    def match_streaminfos(self, *parameters):
        """Match streams given stream_ids or property values.

        See pyxdf.match_streaminfos for matching options.
        """
        return pyxdf.match_streaminfos(self.resolve_streams(), parameters)

    def available_stream_ids(self):
        """Return a list of available stream IDs."""
        streams = RawXdf.resolve_streams(self)
        stream_ids = sorted([stream['stream_id'] for stream in streams])
        return stream_ids

    def load(self, *select_streams, **kwargs):
        """Load XDF data using pyxdf passing all kwargs."""
        if self.loaded():
            raise UserWarning('Streams already loaded.')
        kwargs['verbose'] = self.verbose
        if not select_streams:
            select_streams = None
        try:
            streams, header = pyxdf.load_xdf(
                self.filename,
                select_streams,
                **kwargs,
            )
        except np.linalg.LinAlgError:
            loadable_streams = self._find_loadable_streams(
                select_streams, **kwargs)
            if loadable_streams:
                streams, header = pyxdf.load_xdf(
                    self.filename, loadable_streams, **kwargs)
            else:
                print('No loadable streams!')
                return self

        # Initialise class attributes.
        self._loaded = True
        self._header = header
        self._streams = streams
        return self

    def _find_loadable_streams(self, select_streams, **kwargs):
        if select_streams is None:
            select_streams = self.available_stream_ids()
        elif all([isinstance(elem, int) for elem in select_streams]):
            pass
        elif all([isinstance(elem, dict) for elem in select_streams]):
            select_streams = self.match_streaminfos(*select_streams)

        # Test loading each stream.
        loadable_streams = []
        for i in select_streams:
            try:
                _, _ = pyxdf.load_xdf(self.filename, i, **kwargs)
                loadable_streams.append(i)
            except Exception as exc:
                exc = DataStreamLoadError(i, exc)
                print(exc)
        return loadable_streams

    def loaded(self):
        """Test if a file has been loaded."""
        return self._loaded

    def get_header(self):
        """Return the raw header info dictionary."""
        self._assert_loaded()
        return self._header['info']

    def num_loaded_streams(self):
        """Return the number of streams currently loaded."""
        self._assert_loaded()
        return len(self._streams)

    def loaded_stream_ids(self):
        """Get IDs for all loaded streams."""
        return sorted([self._get_stream_id(stream)
                       for stream in self._streams])

    def get_streams(self, *stream_ids):
        """Return raw stream data.

        Select streams according to their ID or default all loaded
        streams.
        """
        self._assert_loaded()
        # If no stream_ids are provided return all loaded streams.
        if not stream_ids or set(self.loaded_stream_ids()) == set(stream_ids):
            return self._streams
        self._assert_stream_ids(*stream_ids)
        return [stream for stream_id in stream_ids
                for stream in self._streams
                if self._get_stream_id(stream) == stream_id]

    def collect_stream_data(self, *stream_ids, data_path=None,
                            pop_singleton_lists=False,
                            as_key=False):
        """Extract nested stream data for multiple streams.

        Returns a dictionary {stream_id: data} with number of items
        equal to the number of streams. If no data is available at any
        key in the data path the item value will be None.
        """
        streams = self.get_streams(*stream_ids)
        data = {self._get_stream_id(stream):
                self._get_stream_data(stream, data_path,
                                      as_key=as_key)
                for stream in streams}
        if pop_singleton_lists:
            data = self._pop_singleton_lists(data)
        return data

    def collect_leaf_data(self, data, leaf_data=None):
        """Collect singleton items of metadata in leaf nodes."""
        if leaf_data is None:
            leaf_data = {}
        for key, item in data.items():
            if isinstance(item, dict):
                self.collect_leaf_data(item, leaf_data)
            if isinstance(item, list):
                if len(item) == 1:
                    if isinstance(item[0], str) or item[0] is None:
                        leaf_data[key] = item
                    else:
                        self.collect_leaf_data(item[0], leaf_data)
        return leaf_data

    # Non-public methods.
    def _assert_loaded(self):
        """Assert that data is loaded before continuing."""
        if not self.loaded():
            raise UserWarning(
                'No streams loaded, call load_streams() first.')

    def _assert_stream_ids(self, *stream_ids):
        """Assert that requested streams are loaded before continuing."""
        valid_ids = set(self.loaded_stream_ids()).intersection(
            stream_ids)
        try:
            assert len(valid_ids) == len(stream_ids)
        except AssertionError:
            invalid_ids = list(valid_ids.symmetric_difference(stream_ids))
            raise KeyError(f'Invalid stream IDs: {invalid_ids}') from None

    def _get_stream_data(self, stream, data_path, *, as_key=False):
        """Extract nested stream data at data_path."""
        data = stream
        for key in data_path:
            if data and key in data.keys():
                data = data[key]
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
            else:
                stream_id = self._get_stream_id(stream)
                print(f'Stream {stream_id} does not contain key: {key}.')
                return None
        if as_key:
            data = {as_key: data}
        return data

    def _get_stream_id(self, stream):
        # Get ID for stream.
        return self._get_stream_data(stream, ['info', 'stream_id'])

    def _pop_singleton_lists(self, data):
        # Copy dictionary to avoid modifying in place.
        data = data.copy()
        for key, item in data.items():
            if isinstance(item, list):
                if len(item) == 1:
                    item = item[0]
                else:
                    item = [self._pop_singleton_lists(i) for i in item]
            if isinstance(item, dict):
                item = self._pop_singleton_lists(item)

            data[key] = item
        return data
