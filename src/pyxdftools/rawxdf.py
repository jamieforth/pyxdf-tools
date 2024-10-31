"""Thin wrapper for loading and accessing raw XDF data."""

import numbers
from collections.abc import Sequence
from functools import wraps
from warnings import warn

import numpy as np
import pyxdf
import scipy

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


class RawXdf(BaseXdf, Sequence):
    """Thin wrapper for loading and accessing raw XDF data.

    Provides convenience methods for loading and accessing raw XDF data
    (lists of dictionaries).

    Properties:
        loaded: Boolean indicating if a file has been loaded.
        load_params: Dict of pyxdf_load parameters.
        num_loaded_streams: number of streams currently loaded.
        loaded_stream_ids: IDs for all loaded streams.
    """

    _loaded = False

    def __init__(self, filename, verbose=False):
        """Initialise RawXdf via super class."""
        super().__init__(filename, verbose)

    def __getitem__(self, index):
        """Get stream-ID for each loaded streams."""
        return self.loaded_stream_ids[index]

    def __len__(self):
        """Return the number of loaded streams."""
        return self.num_loaded_streams

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
    @XdfDecorators.loaded
    def num_loaded_streams(self):
        """Return the number of streams currently loaded."""
        return len(self.loaded_stream_ids)

    @property
    @XdfDecorators.loaded
    def load_params(self):
        """Return the parameters used to load the data."""
        return self._load_params

    # Public methods.

    def load(self, *select_streams, **kwargs):
        """Load XDF data using pyxdf passing all kwargs.

        Any pyxdf.load_xdf() kwargs provided will be passed to that
        function. All other kwargs will be passed to parsing methods.
        """
        try:
            self._load(*select_streams, **kwargs)
        except (NoLoadableStreamsError, XdfAlreadyLoadedError) as exc:
            print(exc)
        return self

    @XdfDecorators.loaded
    def unload(self):
        """Free memory - useful for handling large datasets."""
        # Reset class attributes.
        self._loaded = False
        del self._loaded_stream_ids

        # Free memory
        del self._header
        del self._metadata
        del self._channel_metadata
        del self._clock_offsets
        del self._time_series
        del self._time_stamps

    @XdfDecorators.loaded
    def header(self):
        """Return the raw header info dictionary."""
        return self._header

    @XdfDecorators.loaded
    def metadata(self, *stream_ids, with_stream_id=False, flatten=False):
        """Return raw stream metadata.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_stream_id=True.

        Flatten=True will place all leaf-node key-value pairs in nested
        dictionaries within the top-level dictionary.
        """
        data = self._get_stream_data(
            *stream_ids,
            data=self._metadata,
            with_stream_id=with_stream_id,
        )
        if flatten:
            data = self.__flatten(data)
        return data

    @XdfDecorators.loaded
    def channel_metadata(self, *stream_ids, with_stream_id=False):
        """Return raw stream channel metadata.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_stream_id=True.
        """
        if self._channel_metadata:
            return self._get_stream_data(
                *stream_ids,
                data=self._channel_metadata,
                with_stream_id=with_stream_id,
            )

    @XdfDecorators.loaded
    def clock_offsets(self, *stream_ids, with_stream_id=False):
        """Return raw stream clock offsets: time and value.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_offsets,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_series(self, *stream_ids, with_stream_id=False):
        """Return raw stream time-series data.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_series,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_stamps(self, *stream_ids, with_stream_id=False):
        """Return raw stream time-stamp data.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id: data}
        where number of items is equal to the number of streams. Single
        streams are returned as is unless with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_stamps,
            with_stream_id=with_stream_id,
        )

    def data(self, *stream_ids, with_stream_id=True):
        time_stamps = self.time_stamps(*stream_ids, with_stream_id=True)
        time_series = self.time_series(*stream_ids, with_stream_id=True)

        data = {
            stream_id: {'time_stamps': t, 'time_series': ts}
            for ((stream_id, t), ts)
            in zip(time_stamps.items(), time_series.values())
        }

        return data

    def single_or_multi_stream_data(self, data, with_stream_id=False):
        """Return single stream data or dictionary."""
        if len(data) == 1 and not with_stream_id:
            return data[list(data.keys())[0]]
        else:
            return data

    def resample_streams(self, *stream_ids, fs_new):
        """
        Resample multiple XDF streams to a given frequency.

        Based on mneLab:
        https://github.com/cbrnr/mnelab/blob/main/src/mnelab/io/xdf.py.

        Parameters
        ----------
        streams : dict
            A dictionary mapping stream IDs to XDF streams.
        stream_ids : list[int]
            The IDs of the desired streams.
        fs_new : float
            Resampling target frequency in Hz.

        Returns
        -------
        all_time_series : np.ndarray
            Array of shape (n_samples, n_channels) containing raw data. Time intervals where a
            stream has no data contain `np.nan`.
        first_time : float
            Time of the very first sample in seconds.
        """
        start_times = []
        end_times = []
        n_total_chans = 0
        for stream_id, stream in self.time_stamps(
                *stream_ids, with_stream_id=True).items():
            start_times.append(stream[0])
            end_times.append(stream[-1])
            n_total_chans += int(self.metadata(
                stream_id)['channel_count'])
        first_time = min(start_times)
        last_time = max(end_times)

        n_samples = int(np.ceil((last_time - first_time) * fs_new))
        all_time_series = np.full((n_samples, n_total_chans), np.nan)

        col_start = 0
        for stream_id, stream in self.time_stamps(
                *stream_ids, with_stream_id=True).items():
            start_time = stream[0]
            end_time = stream[-1]
            len_new = int(np.ceil((end_time - start_time) * fs_new))

            x_old = self.time_series(stream_id)
            x_new = scipy.signal.resample(x_old, len_new, axis=0)

            row_start = int(
                np.floor((stream[0] - first_time) * fs_new)
            )
            row_end = row_start + x_new.shape[0]
            col_end = col_start + x_new.shape[1]
            all_time_series[row_start:row_end, col_start:col_end] = x_new

            col_start += x_new.shape[1]

        return all_time_series, first_time

    # Non-public methods.

    def _load(self, *select_streams, **kwargs):
        if self.loaded:
            raise XdfAlreadyLoadedError

        # Separate kwargs.
        xdf_kwargs = {k: kwargs[k] for k in
                      kwargs.keys() & pyxdf.load_xdf.__kwdefaults__.keys()}
        parse_kwargs = {k: kwargs[k] for k in
                        kwargs.keys() - pyxdf.load_xdf.__kwdefaults__.keys()}

        if 'verbose' not in xdf_kwargs:
            xdf_kwargs['verbose'] = self.verbose

        if not select_streams:
            select_streams = None

        try:
            streams, header = pyxdf.load_xdf(self.filename, select_streams,
                                             **xdf_kwargs)
        except np.linalg.LinAlgError:
            loadable_streams = self._find_loadable_streams(
                select_streams, **xdf_kwargs)
            if loadable_streams:
                streams, header = pyxdf.load_xdf(self.filename,
                                                 loadable_streams,
                                                 **xdf_kwargs)
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
        self._load_params = {k: v
                             for (k, v) in xdf_kwargs.items()
                             if k not in ['verbose']}

        # Parse XDF into separate structures.
        self._header = self._parse_header(header, **parse_kwargs)
        self._metadata = self._parse_metadata(streams, **parse_kwargs)
        self._channel_metadata = self._parse_channel_metadata(streams,
                                                              **parse_kwargs)
        self._clock_offsets = self._parse_clock_offsets(streams,
                                                        **parse_kwargs)
        self._time_series = self._parse_time_series(streams, **parse_kwargs)
        self._time_stamps = self._parse_time_stamps(streams, **parse_kwargs)

        return self

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
    def _parse_clock_offsets(self, data, **kwargs):
        # Extract clock offsets.
        clock_times = self.__collect_stream_data(
            data=data,
            data_path=['clock_times'],
        )
        clock_values = self.__collect_stream_data(
            data=data,
            data_path=['clock_values'],
        )

        clock_offsets = {
            stream_id: {
                'time': times,
                'value': clock_values[stream_id]
            } for stream_id, times in clock_times.items()
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

    def _get_stream_data(self, *stream_ids, data, with_stream_id):
        if not stream_ids or data.keys() == set(stream_ids):
            pass
        else:
            try:
                self._assert_stream_ids(*stream_ids, data=data)
                data = {stream_id: data[stream_id]
                        for stream_id in stream_ids}
            except KeyError as exc:
                print(exc)
                return None
        return self.single_or_multi_stream_data(data, with_stream_id)

    def _assert_loaded(self):
        """Assert that data is loaded before continuing."""
        if not self.loaded:
            raise XdfNotLoadedError

    def _assert_stream_ids(self, *stream_ids, data):
        """Assert that requested streams are loaded before continuing."""
        unique_ids = self.remove_duplicates(stream_ids)
        valid_ids = set(unique_ids).intersection(data.keys())
        if len(valid_ids) != len(unique_ids):
            invalid_ids = list(valid_ids.symmetric_difference(stream_ids))
            raise KeyError(f'Invalid stream IDs: {invalid_ids}')

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
