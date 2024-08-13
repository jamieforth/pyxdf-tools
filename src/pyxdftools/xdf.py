"""Main Xdf class for working with XDF data."""

import mne
import numpy as np
import pandas as pd

from .constants import microvolts
from .rawxdf import RawXdf, XdfDecorators


class Xdf(RawXdf):
    """Main class for XDF data processing with pandas.

    Provides a pandas-based layer of abstraction over raw XDF data to
    simplify data processing.
    """

    # Data types for XDF metadata.
    _metadata_types = {
        'channel_count': np.int16,
        'nominal_srate': np.float64,
        'v4data_port': np.int32,
        'v4service_port': np.int32,
        'v6data_port': np.int32,
        'v6service_port': np.int32,
        'effective_srate': np.float64,
        'first_timestamp': np.float64,
        'last_timestamp': np.float64,
        'sample_count': np.int64,
    }

    def __init__(self, filename, verbose=False):
        """Initialise XdfData via super class."""
        super().__init__(filename, verbose)

    def resolve_streams(self):
        """Return a DataFrame containing available streams.

        Results are not cached - the data is always read from file.
        """
        streams = pd.DataFrame(super().resolve_streams())
        streams.set_index('stream_id', inplace=True)
        streams.sort_index(inplace=True)
        return streams

    def load(self, *select_streams, channel_scale_field=None,
             channel_name_field=None, **kwargs):
        """Load XDF data from file using pyxdf.load_xdf().

        Any pyxdf.load_xdf() kwargs provided will be passed to that
        function. All other kwargs are assumed to be stream properties
        and will be passed to resolve_streams().
        """
        super().load(*select_streams,
                     channel_scale_field=channel_scale_field,
                     channel_name_field=channel_name_field,
                     **kwargs)
        return self

    @XdfDecorators.loaded
    def metadata(self, *stream_ids, cols=[]):
        """Return stream metadata as a DataFrame.

        Select data for stream_ids or default all loaded streams.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._metadata,
            cols=cols,
        )

    @XdfDecorators.loaded
    def channel_metadata(self, *stream_ids, cols=[], with_stream_id=False):
        """Return channel metadata as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        if self._channel_metadata:
            return self._get_stream_data(
                *stream_ids,
                data=self._channel_metadata,
                cols=cols,
                with_stream_id=with_stream_id,
            )
        else:
            return None

    @XdfDecorators.loaded
    def clock_times(self, *stream_ids, with_stream_id=False):
        """Return clock times as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_times,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def clock_offsets(self, *stream_ids, cols=[], with_stream_id=False):
        """Return clock offset data as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_offsets,
            cols=cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_series(self, *stream_ids, cols=[], with_stream_id=False):
        """Return stream time-series data as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_series,
            cols=cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_stamps(self, *stream_ids, with_stream_id=False):
        """Return stream time-stamps as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._time_stamps,
            with_stream_id=with_stream_id,
        )

    def channel_scalings(self, *stream_ids, channel_scale_field):
        """Return a dictionary of DataFrames with channel scaling values."""
        stream_units = self.channel_metadata(*stream_ids,
                                             cols=channel_scale_field,
                                             with_stream_id=True)
        if stream_units is not None:
            scaling = {stream_id: ch_units.apply(
                lambda units: [1e-6 if u in microvolts else np.nan
                               for u in units])
                       for stream_id, ch_units in stream_units.items()}
            return scaling

    def data(self, *stream_ids, cols=[], with_stream_id=False):
        """Return stream time-series and time-stamps as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        time_series = self.time_series(*stream_ids,
                                       cols=cols,
                                       with_stream_id=True)
        times = self.time_stamps(*stream_ids,
                                 with_stream_id=True)
        ts = {stream_id: ts.join(times[stream_id]).set_index('time_stamp')
              for stream_id, ts in time_series.items()}
        return self.single_or_multi_stream_data(ts, with_stream_id)

    # Non public methods.

    def _parse_header(self, data, **kwargs):
        """Convert raw header into a DataFrame."""
        header = super()._parse_header(data)
        header = pd.Series(header)
        header['datetime'] = pd.to_datetime(header['datetime'])
        return header

    def _parse_metadata(self, data, **kwargs):
        """Parse metadata for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within this instance.

        This method can be implemented by a subclass for custom parsing
        requirements.
        """
        data = super()._parse_metadata(data, flatten=True)
        df = pd.DataFrame(data).T
        df = df.astype(self._metadata_types)
        assert all(df.index == df['stream_id'])
        df.set_index('stream_id', inplace=True)
        return df

    def _parse_channel_metadata(self, data, **kwargs):
        """Parse channel metadata for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Always returns a hierarchical (multiindex) DataFrame, even if
        only a single stream is loaded.
        """
        # Check that data streams have valid channel metadata.
        data = super()._parse_channel_metadata(data)
        data, empty = self._remove_empty_streams(data)
        if empty:
            print(f"""No channel metadata for streams: {' '.join(str(i)
            for i in sorted(list(empty.keys())))}""")
        if not data:
            print('No channel metadata found!')
            return None
        data = self._to_DataFrames(data, 'channel')
        return data

    def _parse_clock_times(self, data, **kwargs):
        """Parse clock times for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Always returns a non-hierarchical DataFrame.
        """
        data = super()._parse_clock_times(data)
        data = self._to_DataFrames(data, 'sample', columns=['time'])
        return data

    def _parse_clock_offsets(self, data, **kwargs):
        """Parse clock offsets for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Always returns a hierarchical (multiindex) DataFrame, even if
        only a single stream is loaded.
        """
        data = super()._parse_clock_offsets(data, pop_singleton_lists=True)
        data = self._to_DataFrames(data, 'samples')
        return data

    def _parse_time_series(self, data, channel_scale_field,
                           channel_name_field, **kwargs):
        """Parse time-series data for all loaded streams into a DataFrame.

        Optionally scales values and sets channel names according to channel
        metadata.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Always returns a hierarchical (multiindex) DataFrame, even if
        only a single stream is loaded.
        """
        data = super()._parse_time_series(data)
        data = self._to_DataFrames(data, 'sample',
                                   col_index_name='channels')

        if channel_scale_field:
            scalings = self.channel_scalings(
                channel_scale_field=channel_scale_field)
            if scalings:
                data = {stream_id: ts * scalings[stream_id].loc[
                    stream_id, channel_scale_field
                ]
                        if stream_id in scalings else ts
                        for stream_id, ts in data.items()}

        if channel_name_field:
            ch_labels = self.channel_metadata(cols=channel_name_field,
                                              with_stream_id=True)
            if ch_labels:
                data = {stream_id: ts.rename(
                    columns=ch_labels[stream_id].loc[:, channel_name_field])
                        if stream_id in ch_labels else ts
                        for stream_id, ts in data.items()}
        return data

    def _parse_time_stamps(self, data, **kwargs):
        """Parse time-stamps for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Always returns a hierarchical (multiindex) DataFrame, even if
        only a single stream is loaded.
        """
        data = super()._parse_time_stamps(data)
        data = self._to_DataFrames(data,
                                   'samples',
                                   columns=['time_stamp'])
        return data

    def _get_stream_data(self, *stream_ids, data, cols=[],
                         with_stream_id=False):
        if isinstance(data, dict):
            data = super()._get_stream_data(*stream_ids, data=data,
                                            with_stream_id=with_stream_id)
        elif isinstance(data, pd.DataFrame):
            if stream_ids and set(self.loaded_stream_ids) != set(stream_ids):
                data = data.loc[list(stream_ids), :]
        else:
            raise ValueError('Data should be a dictionary or DataFrame')
        # Subset data.
        if data is not None and cols:
            if not isinstance(cols, list):
                cols = [cols]
            try:
                self._assert_columns(data, cols)
            except KeyError as exc:
                print(exc)
                return None
            if isinstance(data, pd.DataFrame):
                data = data.loc[:, cols]
            else:
                data = {stream_id: df.loc[:, cols]
                        for stream_id, df in data.items()}
        return data

    # def _get_stream_data_old(self, *stream_ids, data, cols=[],
    #                          with_stream_id=False):
    #     if (
    #             not cols
    #             and (
    #                 not stream_ids
    #                 or set(self.loaded_stream_ids) == set(stream_ids)
    #             )
    #     ):
    #         return data
    #     else:
    #         if cols and not isinstance(cols, list):
    #             cols = [cols]
    #         try:
    #             self._assert_stream_ids(*stream_ids)
    #             self._assert_columns(data, cols)
    #         except KeyError as exc:
    #             print(exc)
    #             return None

    #     stream_ids = list(stream_ids)

    #     if data.columns.nlevels == 1:
    #         if data.columns.name == 'stream_id':
    #             # Data frame with a single-level column index indexed by
    #             # stream-id. Row numbers index observations.
    #             return data.loc[:, stream_ids]
    #         else:
    #             # Data frame with a single-level column index and rows
    #             # indexed by stream-id.
    #             if stream_ids and cols:
    #                 return data.loc[stream_ids, cols]
    #             elif stream_ids and not cols:
    #                 return data.loc[stream_ids, :]
    #             else:
    #                 return data.loc[:, cols]
    #     elif data.columns.nlevels == 2:
    #         # Data frame with a multi-level column index where the first
    #         # level is stream-id and the second the available data
    #         # variables for each stream. Rows numbers index
    #         # observations.
    #         if stream_ids and cols:
    #             return data.loc[:, (stream_ids, cols)]
    #         elif stream_ids and not cols:
    #             return data.loc[:, stream_ids]
    #         else:
    #             return data.loc[:, (slice(None), cols)]
    #     else:
    #         raise UserWarning('Column index must have 1 or 2 levels.')

    def _to_DataFrames(self, data, index_name, col_index_name=None,
                       columns=None):
        # Map a dictionary of {stream-id: data} to a dictionary of
        # {stream-id: DataFrames}.
        data = {stream_id: self._to_df(stream_id, d,
                                       index_name,
                                       col_index_name=col_index_name,
                                       columns=columns)
                for stream_id, d, in data.items()}
        return data

    # def _merge_stream_data(self, data, index_name, *, col_index_name=None,
    #                        col_names=None, with_stream_id=False):
    #     # For single streams return a non-hierarchical DataFrame unless
    #     # with_stream_id=True.
    #     if len(data) == 1 and not with_stream_id:
    #         stream_id = list(data.keys())[0]
    #         data = list(data.values())[0]
    #         data = self._to_df(stream_id, data, index_name, col_index_name,
    #                            col_names)
    #         return data

    #     # Otherwise returns a hierarchical (MultiIndex) DataFrame with a
    #     # two-dimensional column index. Stream ID is the first dimension
    #     # of the multi-level column index.
    #     data = {stream_id: self._to_df(stream_id, d,
    #                                    index_name,
    #                                    col_index_name,
    #                                    col_names)
    #             for stream_id, d, in data.items()}
    #     data = pd.concat(data, axis='columns')
    #     # Set stream_id as the first column index level.
    #     data.columns.set_names('stream_id', level=0, inplace=True)
    #     data.sort_index(axis='columns', level='stream_id',
    #                     sort_remaining=False, inplace=True)
    #     return data

    def _remove_empty_streams(self, data):
        streams = {}
        empty = {}
        for stream_id, d in data.items():
            if d is not None:
                streams[stream_id] = d
            else:
                empty[stream_id] = d
        return streams, empty

    def _to_df(self, stream_id, data, index_name, col_index_name=None,
               columns=None):
        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return None
        df.index.set_names(index_name, inplace=True)
        if col_index_name:
            df.columns.set_names(col_index_name, inplace=True)
        return df

    def _assert_columns(self, data, columns):
        try:
            if isinstance(data, pd.DataFrame):
                assert all([col in data.columns for col in columns])
            else:
                for stream_id, df in data.items():
                    if df.columns.nlevels == 1:
                        assert all([col in df.columns for col in columns])
                    elif df.columns.nlevels == 2:
                        assert all([col in df.columns.levels[1]
                                    for col in columns])
                    else:
                        raise ValueError(
                            'Column index must have 1 or 2 levels.'
                        )
        except AssertionError:
            raise KeyError(f'Invalid columns: {columns}') from None
