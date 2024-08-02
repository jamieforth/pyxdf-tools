"""AntXdf class for working with AntNeuro XDF data."""

from pyxdftools import Xdf


class NlXdf (Xdf):
    """Helper class for working with AntNeuro XDF data files.

    Provides a pandas-based layer of abstraction over raw XDF data to
    simplify data processing.
    """

    hostname_device_mapper = {
        'DESKTOP-3R7C1PH': 'eeg-a',
        'DESKTOP-2TI6RBU': 'eeg-b',
        'DESKTOP-MN7K6RM': 'eeg-c',
        'DESKTOP-URRV98M': 'eeg-d',
        'DESKTOP-DATOEVU': 'eeg-e',
        'TABLET-9I44R1AR': 'eeg-f',
        'DESKTOP-SLAKFQE': 'eeg-g',
        'DESKTOP-6FJTJJN': 'eeg-h',
        'DESKTOP-HDOESKS': 'eeg-i',
        'DESKTOP-LIA3G09': 'eeg-j',
        'DESKTOP-V6779I4': 'eeg-k',
        'DESKTOP-PLV2A7L': 'eeg-l',
        'DESKTOP-SSSOE1L': 'eeg-m',
        'DESKTOP-RM16J67': 'eeg-n',
        'DESKTOP-N2RA68S': 'eeg-o',
        'DESKTOP-S597Q21': 'eeg-p',
        'DESKTOP-OE9298C': 'eeg-q',
        'DESKTOP-MK0GQFM': 'eeg-r',
        'DESKTOP-7GV3RJU': 'eeg-s',
        'DESKTOP-S5A1PPK': 'eeg-t',
        'TABLET-3BS4NTP2': 'eeg-u',
        'DESKTOP-QG4CNEV': 'eeg-v',
        'TABLET-STDTE3Q6': 'eeg-w',
        'DESKTOP-T3RKRMH': 'eeg-x',
        'CGS-PCD-26098': 'marker',
    }

    metadata_mapper = {
        'type': {
            'EEG': 'eeg',
        },
    }

    channel_metadata_mapper = {
        'label': {
            '0': 'Fp1',
            '1': 'Fpz',
            '2': 'Fp2',
            '3': 'F7',
            '4': 'F3',
            '5': 'Fz',
            '6': 'F4',
            '7': 'F8',
            '8': 'FC5',
            '9': 'FC1',
            '10': 'FC2',
            '11': 'FC6',
            '12': 'M1',
            '13': 'T7',
            '14': 'C3',
            '15': 'Cz',
            '16': 'C4',
            '17': 'T8',
            '18': 'M2',
            '19': 'CP5',
            '20': 'CP1',
            '21': 'CP2',
            '22': 'CP6',
            '23': 'P7',
            '24': 'P3',
            '25': 'Pz',
            '26': 'P4',
            '27': 'P8',
            '28': 'POz',
            '29': 'O1',
            '30': 'Oz',
            '31': 'O2',
            '32': 'CPz',  # EEG 101 with 34 channels?
            '67': 'CPz',
            '33': 'trigger',
            '34': 'counter',
        },
        'type': {
            'ref': 'eeg',
            'aux': 'misc',
            'bip': 'misc',
            'trigger': 'misc',
            'counter': 'misc',
            'trg': 'stim'
        },
    }

    def load(
            self,
            *select_streams,
            channel_scale_field='unit',
            channel_name_field='label',
            synchronize_clocks=True,
            dejitter_timestamps=True,
            handle_clock_resets=False,
            **kwargs):
        """Load XDF data from file using pyxdf.load_xdf().

        Apply custom defaults for Neurolive analysis.
        """
        super().load(*select_streams,
                     channel_scale_field=channel_scale_field,
                     channel_name_field=channel_name_field,
                     synchronize_clocks=synchronize_clocks,
                     dejitter_timestamps=dejitter_timestamps,
                     handle_clock_resets=handle_clock_resets,
                     **kwargs)
        return self

    def _parse_metadata(self, data, **kwargs):
        """Rename AntNeuro stream types when parsing metadata."""
        df = super()._parse_metadata(data, **kwargs)
        df.replace(self.metadata_mapper, inplace=True)

        # Create nl_id from hostname and set as index.
        df['nl_id'] = df['hostname']
        df.replace({
            'nl_id': self.hostname_device_mapper
        }, inplace=True)
        df.reset_index(inplace=True)
        df.set_index('nl_id', inplace=True)
        return df

    def _parse_channel_metadata(self, data, **kwargs):
        """Rename AntNeuro channels when parsing metadata."""
        data = super()._parse_channel_metadata(data, **kwargs)

        if data is not None:
            for df in data.values():
                df.replace(self.channel_metadata_mapper,
                           inplace=True)
        return data
