"""Thin wrapper for loading and accessing raw XDF data."""

import numbers
from functools import wraps
from warnings import warn

import numpy as np
import pyxdf

from .basexdf import BaseXdf
from .errors import (NoLoadableStreamsError, XdfAlreadyLoadedError,
                     XdfNotLoadedError, XdfStreamLoadError,
                     XdfStreamParseError)


class XdfDecorators:
    """Internal class providing decorator functions."""

    @staticmethod
    def loaded(f):
        """Decorate loading methods for error handling."""
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                self._assert_loaded()
                return f(self, *args, **kwargs)
            except XdfNotLoadedError as exc:
                print(exc)
        return wrapper

    @staticmethod
    def parse(f):
        """Decorate parsing methods for error handling."""
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except Exception as exc:
                raise XdfStreamParseError(exc)
        return wrapper


class RawXdf(BaseXdf):
    """Thin wrapper for loading and accessing raw XDF data.

    Provides convenience methods for loading and accessing raw XDF data
    (lists of dictionaries).

    Properties:
        loaded: Boolean indicating if a file has been loaded.
        num_loaded_streams: number of streams currently loaded.
        loaded_stream_ids: IDs for all loaded streams.
    """

    _loaded = False

    def __init__(self, filename, verbose=False):
        """Initialise RawXdf via super class."""
        super().__init__(filename, verbose)

    # Properties

    @property
    def loaded(self):
        """Test if a file has been loaded."""
        return self._loaded

    @property
    @XdfDecorators.loaded
    def loaded_stream_ids(self):
        """Get IDs for all loaded streams."""
        return self._loaded_stream_ids

    @property
    def num_loaded_streams(self):
        """Return the number of streams currently loaded."""
        return len(self.loaded_stream_ids)

    # Public methods.

    def load(self, *select_streams, **kwargs):
        """Load XDF data using pyxdf passing all kwargs.

        Any pyxdf.load_xdf() kwargs provided will be passed to that
        function. All other kwargs will be passed to parsing methods.
        """
        # Separate kwargs.
        xdf_kwargs = {k: kwargs[k] for k in
                      kwargs.keys() & pyxdf.load_xdf.__kwdefaults__.keys()}
        parse_kwargs = {k: kwargs[k] for k in
                        kwargs.keys() - pyxdf.load_xdf.__kwdefaults__.keys()}

        try:
            streams, header = self._load(*select_streams, **xdf_kwargs)
        except (NoLoadableStreamsError, XdfAlreadyLoadedError) as exc:
            print(exc)
            return self

        # Parse XDF into separate structures.
        self._header = self._parse_header(header, **parse_kwargs)
        self._metadata = self._parse_metadata(streams, **parse_kwargs)
        self._channel_metadata = self._parse_channel_metadata(streams,
                                                              **parse_kwargs)
        self._clock_times = self._parse_clock_times(streams, **parse_kwargs)
        self._clock_offsets = self._parse_clock_offsets(streams,
                                                        **parse_kwargs)
        self._time_series = self._parse_time_series(streams, **parse_kwargs)
        self._time_stamps = self._parse_time_stamps(streams, **parse_kwargs)
        return self

    @XdfDecorators.loaded
    def header(self):
        """Return the raw header info dictionary."""
        return self._header

    @XdfDecorators.loaded
    def metadata(self, *stream_ids, with_source_id=False, flatten=False):
        """Return raw stream metadata.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.

        Flatten=True will place all leaf-node key-value pairs in nested
        dictionaries within the top-level dictionary.
        """
        data = self._get_stream_data(
            *stream_ids,
            data=self._metadata,
            with_source_id=with_source_id,
        )
        if flatten:
            data = self.__flatten(data)
        return data

    @XdfDecorators.loaded
    def channel_metadata(self, *stream_ids, with_source_id=False):
        """Return raw stream channel metadata.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.
        """
        if self._channel_metadata:
            return self._get_stream_data(
                *stream_ids,
                data=self._channel_metadata,
                with_source_id=with_source_id,
            )

    @XdfDecorators.loaded
    def clock_times(self, *stream_ids, with_source_id=False):
        """Return raw stream clock times.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_times,
            with_source_id=with_source_id,
        )

    @XdfDecorators.loaded
    def clock_offsets(self, *stream_ids, with_source_id=False):
        """Return raw stream clock offsets.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_offsets,
            with_source_id=with_source_id,
        )

    @XdfDecorators.loaded
    def time_series(self, *stream_ids, with_source_id=False):
        """Return raw stream time-series data.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_series,
            with_source_id=with_source_id,
        )

    @XdfDecorators.loaded
    def time_stamps(self, *stream_ids, with_source_id=False):
        """Return raw stream time-stamp data.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_source_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_stamps,
            with_source_id=with_source_id,
        )

    def single_or_multi_stream_data(self, data, with_source_id=False):
        """Return single stream data or dictionary."""
        if len(data) == 1 and not with_source_id:
            return data[list(data.keys())[0]]
        else:
            return data

    # Non-public methods.

    def _load(self, *select_streams, **kwargs):
        if self.loaded:
            raise XdfAlreadyLoadedError
        kwargs['verbose'] = self.verbose
        if not select_streams:
            select_streams = None
        try:
            streams, header = pyxdf.load_xdf(self.filename, select_streams,
                                             **kwargs)
        except np.linalg.LinAlgError:
            loadable_streams = self._find_loadable_streams(
                select_streams, **kwargs)
            if loadable_streams:
                streams, header = pyxdf.load_xdf(self.filename,
                                                 loadable_streams, **kwargs)
            else:
                raise NoLoadableStreamsError(select_streams)

        # Store stream data as a dictionary sorted by stream-id.
        streams.sort(key=self.__get_stream_id)
        stream_ids = [self.__get_stream_id(stream)
                      for stream in streams]
        streams = dict(zip(stream_ids, streams))

        # Initialise class attributes.
        self._loaded_stream_ids = stream_ids
        self._loaded = True
        return streams, header

    def _find_loadable_streams(self, select_streams=None, **kwargs):
        if select_streams is None:
            select_streams = self.available_stream_ids
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
                exc = XdfStreamLoadError(i, exc)
                print(exc)
        return loadable_streams

    # Parsing methods.

    # These methods are called when XDF data is loaded and the returned
    # data is cached within the instance. Sub-classes can override
    # these methods for custom parsing requirements.

    @XdfDecorators.parse
    def _parse_header(self, data, **kwargs):
        # Remove unnecessary list objects.
        header = self.__pop_singleton_lists(data['info'])
        return header

    @XdfDecorators.parse
    def _parse_metadata(self, data, flatten=False, pop_singleton_lists=True,
                        **kwargs):
        # Merge together 'info' and 'footer' metadata, excluding
        # specific list data structures, e.g. channel metadata and clock
        # offset data.
        metadata = self.__collect_stream_data(
            data=data,
            data_path=['info'],
            exclude=['channels'],
            flatten=flatten,
            pop_singleton_lists=pop_singleton_lists,
        )
        footer = self.__collect_stream_data(
            data=data,
            data_path=['footer', 'info'],
            exclude=['clock_offsets'],
            flatten=flatten,
            pop_singleton_lists=pop_singleton_lists,
        )

        if metadata.keys() != footer.keys():
            warn('Mismatched metadata and footer streams. \n'
                 f'Metadata streams: {list(metadata.keys())}, '
                 f'Footer streams: {list(footer.keys())}')

        metadata = {
            stream_id: {
                **metadata[stream_id], **footer[stream_id]
            }
            for stream_id in metadata
        }
        return metadata

    @XdfDecorators.parse
    def _parse_channel_metadata(self, data, pop_singleton_lists=True,
                                **kwargs):
        # Extract channel metadata from stream metadata.
        ch_metadata = self.__collect_stream_data(
            data=data,
            data_path=['info', 'desc', 'channels', 'channel'],
            pop_singleton_lists=pop_singleton_lists,
            allow_none=True,
        )
        return ch_metadata

    @XdfDecorators.parse
    def _parse_clock_times(self, data, pop_singleton_lists=False, **kwargs):
        # Extract clock times from stream data.
        clock_times = self.__collect_stream_data(
            data=data,
            data_path=['clock_times'],
            pop_singleton_lists=pop_singleton_lists,
        )
        return clock_times

    @XdfDecorators.parse
    def _parse_clock_offsets(self, data, pop_singleton_lists=True, **kwargs):
        # Extract clock offsets from footer data.
        clock_offsets = self.__collect_stream_data(
            data=data,
            data_path=['footer', 'info', 'clock_offsets', 'offset'],
            pop_singleton_lists=pop_singleton_lists,
        )
        # Coerce strings to floats.
        clock_offsets = {
            stream_id: [{key: [float(f) for f in value]
                         if isinstance(value, list) else float(value)
                         for key, value in props.items()}
                        for props in data]
            for stream_id, data in clock_offsets.items()
        }
        return clock_offsets

    @XdfDecorators.parse
    def _parse_time_series(self, data, **kwargs):
        # Extract time series data from stream data, e.g. EEG.
        time_series = self.__collect_stream_data(
            data=data,
            data_path='time_series',
        )
        return time_series

    @XdfDecorators.parse
    def _parse_time_stamps(self, data, **kwargs):
        # Extract time stamps from stream data, e.g. time stamps
        # corresponding to EEG samples.
        time_stamps = self.__collect_stream_data(
            data=data,
            data_path='time_stamps',
        )
        return time_stamps

    def _get_stream_data(self, *stream_ids, data, with_source_id):
        if not stream_ids or set(self.loaded_stream_ids) == set(stream_ids):
            pass
        else:
            try:
                self._assert_stream_ids(*stream_ids)
                data = {stream_id: data[stream_id]
                        for stream_id in stream_ids}
            except KeyError as exc:
                print(exc)
                return None
        return self.single_or_multi_stream_data(data, with_source_id)

    def _assert_loaded(self):
        """Assert that data is loaded before continuing."""
        if not self.loaded:
            raise XdfNotLoadedError

    def _assert_stream_ids(self, *stream_ids):
        """Assert that requested streams are loaded before continuing."""
        valid_ids = set(self.loaded_stream_ids).intersection(
            stream_ids)
        try:
            assert len(valid_ids) == len(stream_ids)
        except AssertionError:
            invalid_ids = list(valid_ids.symmetric_difference(stream_ids))
            raise KeyError(f'Invalid stream IDs: {invalid_ids}') from None

    # Name-mangled private methods to be used only by this class.

    def __get_stream_id(self, stream):
        # Get ID from stream data.
        return self.__find_data_at_path(stream, ['info', 'stream_id'])

    def __collect_stream_data(self, data, data_path, exclude=None,
                              pop_singleton_lists=False, flatten=False,
                              allow_none=False):
        """Extract data from nested stream dictionaries at the data_path.

        Stream data is always returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams.

        If no data is available at any key in the data path the item
        value will be None.
        """
        if not isinstance(data_path, list):
            data_path = [data_path]
        data = {stream_id:
                self.__find_data_at_path(
                    stream,
                    data_path,
                    allow_none=allow_none,
                )
                for stream_id, stream in data.items()}
        if exclude:
            data = self.__filter_stream_data(data, exclude)
        if flatten:
            data = self.__flatten(data)
        if pop_singleton_lists:
            data = self.__pop_singleton_lists(data)
        return data

    def __find_data_at_path(self, stream, data_path, allow_none=False):
        """Extract nested stream data at data_path."""
        data = stream
        for key in data_path:
            if data and key in data.keys():
                data = data[key]
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
            else:
                stream_id = self.__get_stream_id(stream)
                if allow_none:
                    return None
                else:
                    raise KeyError(
                        f'Stream {stream_id} does not contain key {key} '
                        f'at path {data_path}'
                    )
        return data

    def __filter_stream_data(self, data, exclude):
        # Allow exclude to be provided as a single value or list.
        if not isinstance(exclude, list):
            exclude = [exclude]

        if isinstance(data, dict):
            return {k: self.__filter_stream_data(v, exclude)
                    for k, v in data.items() if k not in exclude}
        elif isinstance(data, list):
            return [self.__filter_stream_data(item, exclude) for item in data]
        else:
            return data

    def __pop_singleton_lists(self, data):
        if isinstance(data, dict):
            return {k: self.__pop_singleton_lists(v)
                    for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) == 1:
                return self.__pop_singleton_lists(data[0])
            else:
                return [self.__pop_singleton_lists(item) for item in data]
        else:
            return data

    def __flatten(self, data):
        data = {stream_id: self.__collect_leaf_data(stream)
                for stream_id, stream in data.items()}
        return data

    def __collect_leaf_data(self, data, leaf_data=None):
        if leaf_data is None:
            leaf_data = {}
        for key, item in data.items():
            if isinstance(item, (numbers.Number, str)):
                if key not in leaf_data:
                    leaf_data[key] = item
                else:
                    raise KeyError(f'Duplicate key {key}.')
            if isinstance(item, dict):
                self.__collect_leaf_data(item, leaf_data)
            if isinstance(item, list):
                if len(item) == 1:
                    if isinstance(item[0], (numbers.Number, str)):
                        if key not in leaf_data:
                            leaf_data[key] = item
                        else:
                            raise KeyError(f'Duplicate key {key}.')
                    elif isinstance(item[0], dict):
                        self.__collect_leaf_data(item[0], leaf_data)
        return leaf_data
