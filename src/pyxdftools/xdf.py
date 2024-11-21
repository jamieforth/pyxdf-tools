"""Main Xdf class for working with XDF data."""

from warnings import warn

import mne
import numpy as np
import pandas as pd
import scipy

from .constants import microvolts
from .errors import NoLoadableStreamsError, XdfAlreadyLoadedError
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
    }

    _footer_types = {
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
        and will be passed to parsing methods.
        """
        try:
            self._load(*select_streams,
                       channel_scale_field=channel_scale_field,
                       channel_name_field=channel_name_field,
                       **kwargs)
        except (NoLoadableStreamsError, XdfAlreadyLoadedError) as exc:
            print(exc)
        return self

    @XdfDecorators.loaded
    def metadata(self, *stream_ids, exclude=[], cols=None,
                 ignore_missing_cols=False):
        """Return stream metadata as a DataFrame.

        Select data for stream_ids or default all loaded streams.
        """
        return self._get_stream_data(
            *stream_ids,
            data=self._metadata,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
        )

    @XdfDecorators.loaded
    def channel_metadata(self, *stream_ids, exclude=[], cols=None,
                         ignore_missing_cols=False, with_stream_id=False,):
        """Return channel metadata as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        if not self._channel_metadata:
            print('No channel metadata.')
            return None
        channel_metadata = self._get_stream_data(
            *stream_ids,
            data=self._channel_metadata,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )
        return channel_metadata

    @XdfDecorators.loaded
    def footer(self, *stream_ids, exclude=[], cols=None,
               ignore_missing_cols=False):
        """Return stream footer metadata as a DataFrame.

        Select data for stream_ids or default all loaded streams.
        """
        if self._footer is None:
            print('No footer data.')
            return None
        return self._get_stream_data(
            *stream_ids,
            data=self._footer,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
        )

    @XdfDecorators.loaded
    def clock_offsets(self, *stream_ids, exclude=[], cols=None,
                      ignore_missing_cols=False, with_stream_id=False):
        """Return clock offset data as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        if not self._clock_offsets:
            print('No clock-offset data.')
            return None
        return self._get_stream_data(
            *stream_ids,
            data=self._clock_offsets,
            exclude=exclude,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_series(self, *stream_ids, exclude=[], cols=None,
                    ignore_missing_cols=False, with_stream_id=False):
        """Return stream time-series data as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        if not self._time_series:
            print('No time-series data.')
            return None
        return self._get_stream_data(
            *stream_ids,
            exclude=exclude,
            data=self._time_series,
            cols=cols,
            ignore_missing_cols=ignore_missing_cols,
            with_stream_id=with_stream_id,
        )

    @XdfDecorators.loaded
    def time_stamps(self, *stream_ids, exclude=[], with_stream_id=False):
        """Return stream time-stamps as a DataFrame.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        if not self._time_stamps:
            print('No time-stamp data.')
            return None
        return self._get_stream_data(
            *stream_ids,
            exclude=exclude,
            data=self._time_stamps,
            with_stream_id=with_stream_id,
        )

    def channel_scalings(self, *stream_ids, channel_scale_field):
        """Return a dictionary of DataFrames with channel scaling values."""
        stream_units = self.channel_metadata(*stream_ids,
                                             cols=channel_scale_field,
                                             ignore_missing_cols=True,
                                             with_stream_id=True)
        if stream_units is not None:
            scaling = {stream_id: ch_units.apply(
                lambda units: [1e-6 if u in microvolts else 1
                               for u in units])
                       for stream_id, ch_units in stream_units.items()}
            return scaling

    def data(self, *stream_ids, exclude=[], cols=None,
             ignore_missing_cols=False, with_stream_id=False,
             as_single_df=False):
        """Return stream time-series and time-stamps as DataFrames.

        Select data for stream_ids or default all loaded streams.

        Multiple streams are returned as a dictionary {stream_id:
        DataFrame} where number of items is equal to the number of
        streams. Single streams are returned as is unless
        with_stream_id=True.
        """
        time_series = self.time_series(*stream_ids,
                                       exclude=exclude,
                                       cols=cols,
                                       ignore_missing_cols=ignore_missing_cols,
                                       with_stream_id=True)
        if not time_series:
            return None
        time_stamps = self.time_stamps(*stream_ids,
                                       exclude=exclude,
                                       with_stream_id=True)
        if not time_stamps:
            return None

        ts = {stream_id:
              ts.join(time_stamps[stream_id]).set_index('time_stamp')
              for stream_id, ts in time_series.items()}
        if as_single_df:
            ts = pd.concat(ts, axis=1).sort_index()
            return ts
        else:
            return self.single_or_multi_stream_data(ts, with_stream_id)

    def time_stamp_summary(self, *stream_ids, exclude=[]):
        """Generate a summary of loaded time-stamp data."""
        time_stamps = self.time_stamps(*stream_ids,
                                       exclude=exclude,
                                       with_stream_id=True)
        if not time_stamps:
            return None
        data = {}
        for stream_id, ts in time_stamps.items():
            data[stream_id] = {
                'sample_count': len(ts),
                'first_timestamp': ts.iloc[0, 0],
                'last_timestamp': ts.iloc[-1, 0],
            }
        data = pd.DataFrame(data).T
        data.index.rename('stream_id', inplace=True)
        data['sample_count'] = data['sample_count'].astype(int)
        data['duration_sec'] = (data['last_timestamp'] -
                                data['first_timestamp'])
        data['duration_min'] = data['duration_sec'] / 60
        data['effective_srate'] = 1 / (data['duration_sec'] /
                                       data['sample_count'])
        data.attrs.update({'load_params': self.load_params})
        return data

    def time_stamp_intervals(self, *stream_ids, exclude=[],
                             with_stream_id=True):
        """Return time-stamp intervals for each stream."""
        time_stamps = self.time_stamps(*stream_ids,
                                       exclude=exclude,
                                       with_stream_id=True)
        if time_stamps is None:
            return None
        data = {}
        for stream_id, ts in time_stamps.items():
            data[stream_id] = ts['time_stamp'].diff()
        data = pd.DataFrame(data)
        data.attrs.update({'load_params': self.load_params})
        return data

    def resample(self, *stream_ids, fs_new, exclude=[], cols=None,
                 ignore_missing_cols=False):
        """
        Resample multiple XDF streams to a given frequency.

        Based on mneLab:
        https://github.com/cbrnr/mnelab/blob/main/src/mnelab/io/xdf.py.

        Parameters
        ----------
        stream_ids : list[int]
            The IDs of the desired streams.
        fs_new : float
            Resampling target frequency in Hz.

        Returns
        -------
        all_time_series : np.ndarray
            Array of shape (n_samples, n_channels) containing raw data. Time
            intervals where a stream has no data contain `np.nan`.
        first_time : float
            Time of the very first sample in seconds.
        """
        if cols is not None and not isinstance(cols, list):
            cols = [cols]
        start_times = []
        end_times = []
        n_total_chans = 0
        for stream_id, time_stamps in self.time_stamps(
                *stream_ids,
                exclude=exclude,
                with_stream_id=True).items():
            start_times.append(time_stamps.iloc[0].item())
            end_times.append(time_stamps.iloc[-1].item())
            if cols:
                n_total_chans += len(cols)
            else:
                n_total_chans += self.metadata(
                    stream_id)['channel_count'].iloc[0]
        first_time = min(start_times)
        last_time = max(end_times)

        n_samples = int(np.ceil((last_time - first_time) * fs_new))
        all_resampled = {}

        for stream_id, time_stamps in self.time_stamps(
                *stream_ids,
                exclude=exclude,
                with_stream_id=True).items():
            start_time = time_stamps.iloc[0].item()
            end_time = time_stamps.iloc[-1].item()
            len_new = int(np.ceil((end_time - start_time) * fs_new))

            x_old = self.time_series(stream_id,
                                     exclude=exclude,
                                     cols=cols,
                                     ignore_missing_cols=ignore_missing_cols)
            x_new = scipy.signal.resample(x_old, len_new, axis=0)
            resampled = np.full((n_samples, x_new.shape[1]), np.nan)

            row_start = int(
                np.floor((time_stamps.iloc[0].item() - first_time) * fs_new)
            )
            row_end = row_start + x_new.shape[0]
            col_end = x_new.shape[1]
            resampled[row_start:row_end, 0:col_end] = x_new
            resampled = pd.DataFrame(resampled, columns=x_old.columns)
            all_resampled[stream_id] = resampled

        all_resampled = pd.concat(all_resampled, axis='columns')
        all_resampled.columns.rename('stream', level=0, inplace=True)
        all_resampled.attrs.update({'load_params': self.load_params})
        return all_resampled, first_time

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

        Returns a DataFrame.
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

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        # Check that data streams have valid channel metadata.
        data = super()._parse_channel_metadata(data)
        data = self._check_empty_streams(data, 'metadata')
        if not data:
            return None
        # Handle streams with only a single channel.
        data = {k: [v] if isinstance(v, dict) else v
                for k, v in  data.items()}
        data = self._to_DataFrames(data, 'channel')
        return data

    def _parse_footer(self, data, **kwargs):
        """Parse footer metadata for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_footer(data)
        data = self._check_empty_streams(data, 'footer')
        if not data:
            return None
        df = pd.DataFrame(data).T
        df = df.astype(self._footer_types)
        return df

    def _parse_clock_offsets(self, data, **kwargs):
        """Parse clock offsets for all loaded streams into a DataFrame.

        Called automatically when XDF data is loaded and the returned
        DataFrame is cached within the instance.

        This method can be implemented by a subclass for custom parsing
        requirements.

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_clock_offsets(data)
        data = self._check_empty_streams(data, 'clock-offsets')
        if not data:
            return None
        data = self._to_DataFrames(data, 'period')
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

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_time_series(data)
        data = self._check_empty_streams(data, 'time-series')
        if not data:
            return None
        data = self._to_DataFrames(data, 'sample',
                                   col_index_name='channel')

        if channel_scale_field:
            scalings = self.channel_scalings(
                channel_scale_field=channel_scale_field)
            if scalings:
                data = {stream_id: ts * scalings[stream_id][channel_scale_field]
                        if (stream_id in scalings
                            and not (scalings[stream_id] == 1).all().item())
                        else ts
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

        Returns a dictionary {stream_id: DataFrame} where number of
        items is equal to the number of streams.
        """
        data = super()._parse_time_stamps(data)
        data = self._check_empty_streams(data, 'time-stamps')
        if not data:
            return None
        data = self._to_DataFrames(data,
                                   'sample',
                                   columns=['time_stamp'])
        return data

    def _get_stream_data(self, *stream_ids, data, exclude=[], cols=None,
                         ignore_missing_cols=False, with_stream_id=False):
        if isinstance(data, dict):
            data = super()._get_stream_data(*stream_ids,
                                            data=data,
                                            exclude=exclude,
                                            with_stream_id=with_stream_id)
        elif isinstance(data, pd.DataFrame):
            if not isinstance(exclude, list):
                exclude = [exclude]
            if stream_ids:
                stream_ids = set(stream_ids)
                stream_ids = stream_ids - set(exclude)
                if set(self.loaded_stream_ids) != stream_ids:
                    data = data.loc[list(stream_ids), :]
            elif len(exclude) > 0:
                data = data.loc[~data.index.isin(exclude)]
        else:
            raise ValueError('Data should be a dictionary or DataFrame')
        # Subset data columns.
        if data is not None and cols is not None:
            if not isinstance(cols, list):
                cols = [cols]
            if isinstance(data, dict):
                subset = {}
                for stream_id in data.keys():
                    df_cols = self._check_columns(data[stream_id],
                                                  cols,
                                                  ignore_missing_cols)
                    if df_cols:
                        subset[stream_id] = data[stream_id].loc[:, df_cols]
                data = subset
            else:
                df_cols = self._check_columns(data, cols, ignore_missing_cols)
                data = data.loc[:, df_cols]
        return data

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

    def _to_df(self, stream_id, data, index_name, col_index_name=None,
               columns=None):
        df = pd.DataFrame(data, columns=columns)
        df.index.set_names(index_name, inplace=True)
        if col_index_name:
            df.columns.set_names(col_index_name, inplace=True)
        df.attrs.update({'load_params': self.load_params})
        return df

    def _remove_empty_streams(self, data):
        streams = {}
        empty = {}
        for stream_id, d in data.items():
            if (d is None
                or (isinstance(d, list)
                    and len(d) == 0)
                or (isinstance(d, dict)
                    and all([len(x) == 0 for x in  d.values()]))
                or (isinstance(d, np.ndarray)
                    and d.size == 0)):
                empty[stream_id] = d
            else:
                streams[stream_id] = d
        return streams, empty

    def _check_empty_streams(self, data, name):
        data, empty = self._remove_empty_streams(data)
        if empty and self.verbose:
            print(f"""No {name} for streams: {' '.join(str(i)
            for i in sorted(list(empty.keys())))}""")
        if not data:
            print(f'No {name} found!')
            return None
        return data

    def _check_columns(self, df, columns, ignore_missing):
        columns = self.remove_duplicates(columns)
        valid_cols = [col for col in columns
                      if col in df.columns]
        if not ignore_missing and len(valid_cols) != len(columns):
            invalid_cols = set(columns).difference(df.columns)
            raise KeyError(f'Invalid columns: {invalid_cols}')
        return valid_cols
