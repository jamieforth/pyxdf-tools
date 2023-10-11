# pyxdf-tools

## Install
### Install into an existing environment

```
pip install -e git+https://github.com/jamieforth/pyxdf-tools.git#egg=pyxdftools
```

### Create a new environment from this repo using `pipenv`

```
git clone https://github.com/jamieforth/pyxdf-tools.git
cd pyxdf-tools
pipenv install
```

## Usage

### Inspecting streams

```
from pyxdftools import XdfData

xdf_data_path = '<filename>.xdf'

# Inspect all available streams.
XdfData(xdf_data_path).resolve_streams()

# Inspect streams by ID.
XdfData(xdf_data_path).resolve_streams(stream_id=[1,2])

# Inspect streams by properties (accepts multiple keyword args).
XdfData(xdf_data_path).resolve_streams(type='eeg')
```

Returns a pandas DataFrame.

| stream\_id | name          | type | source\_id        | created\_at | uid                                  | session\_id | hostname | channel\_count | channel\_format | nominal\_srate |
|------------|---------------|------|-------------------|-------------|--------------------------------------|-------------|----------|----------------|-----------------|----------------|
| 1          | Test stream 0 | eeg  | simulate.py:78559 | 80860.5     | 209ecbcf-08f6-414b-b4ab-6eaa9484174e | default     | kassia   | 2              | float32         | 1              |
| 2          | Test stream 1 | eeg  | simulate.py:78559 | 80860.5     | 79a34624-1171-4988-96fd-cb43b79d7fa4 | default     | kassia   | 2              | float32         | 1              |

### Loading data

```
# Load all streams.
xdf = XdfData(xdf_data_path).load()

# Load subset of streams by ID.
xdf = XdfData(xdf_data_path).load(1, 2)

# Load streams matching properties.
xdf = XdfData(xdf_data_path).load(type='eeg')
```

### Inspecting metadata for loaded streams

#### Stream metadata

```
xdf.metadata()     # Accepts optional stream IDs
```

Returns a pandas DataFrame including all stream header and footer
metadata.

| stream\_id | name          | type | channel\_count | channel\_format | source\_id        | nominal\_srate | version | created\_at | uid                                  | session\_id | hostname | v4address | v4data\_port | v4service\_port | v6address | v6data\_port | v6service\_port | stream\_id | effective\_srate | manufacturer | first\_timestamp | last\_timestamp | sample\_count |
|------------|---------------|------|----------------|-----------------|-------------------|----------------|---------|-------------|--------------------------------------|-------------|----------|-----------|--------------|-----------------|-----------|--------------|-----------------|------------|------------------|--------------|------------------|-----------------|---------------|
| 1          | Test stream 0 | eeg  | 2              | float32         | simulate.py:78559 | 1              | 1.1     | 80860.5     | 209ecbcf-08f6-414b-b4ab-6eaa9484174e | default     | kassia   |           | 16573        | 16596           |           | 16575        | 16598           | 1          | 0.999821         | Neurolive    | 80863.5          | 80871.5         | 8             |
| 2          | Test stream 1 | eeg  | 2              | float32         | simulate.py:78559 | 1              | 1.1     | 80860.5     | 79a34624-1171-4988-96fd-cb43b79d7fa4 | default     | kassia   |           | 16572        | 16597           |           | 16574        | 16599           | 2          | 0.999807         | Neurolive    | 80863.5          | 80871.5         | 8             |

#### Data channel metadata

```
xdf.channel_metadata()     # Accepts optional stream IDs
```

Returns a pandas DataFrame including channel metadata. With no stream
ID arguments returns channel metadata for all loaded streams. 

Multiple streams are returned as a `DataFrame` with a two-dimensional
(`MultiIndex`) column index where `stream_id` is the first dimension.


| channel | (1, &rsquo;label&rsquo;) | (1, &rsquo;unit&rsquo;) | (1, &rsquo;type&rsquo;) | (2, &rsquo;label&rsquo;) | (2, &rsquo;unit&rsquo;) | (2, &rsquo;type&rsquo;) |
|---------|--------------------------|-------------------------|-------------------------|--------------------------|-------------------------|-------------------------|
| 0       | ch:0                     | V                       | eeg                     | ch:0                     | V                       | eeg                     |
| 1       | ch:1                     | V                       | eeg                     | ch:1                     | V                       | eeg                     |

### Stream data as pandas data frames

```
# Get stream time-series data.
df = xdf.time_series()     # Accepts optional stream IDs
```

| sample | (1, &rsquo;ch:0&rsquo;) | (1, &rsquo;ch:1&rsquo;) | (2, &rsquo;ch:0&rsquo;) | (2, &rsquo;ch:1&rsquo;) |
|--------|-------------------------|-------------------------|-------------------------|-------------------------|
| 0      | 2                       | 2                       | 2                       | 2                       |
| 1      | 3                       | 3                       | 3                       | 3                       |
| 2      | 4                       | 4                       | 4                       | 4                       |
| 3      | 5                       | 5                       | 5                       | 5                       |
| 4      | 6                       | 6                       | 6                       | 6                       |
| 5      | 7                       | 7                       | 7                       | 7                       |
| 6      | 8                       | 8                       | 8                       | 8                       |
| 7      | 9                       | 9                       | 9                       | 9                       |
| 8      | 5                       | 5                       | 5                       | 5                       |

```
# Get stream time-stamps.
df = xdf.time_stamps()     # Accepts optional stream IDs
```

| sample | (1, &rsquo;time\_stamp&rsquo;) | (2, &rsquo;time\_stamp&rsquo;) |
|--------|--------------------------------|--------------------------------|
| 0      | 80863.5                        | 80863.5                        |
| 1      | 80864.5                        | 80864.5                        |
| 2      | 80865.5                        | 80865.5                        |
| 3      | 80866.5                        | 80866.5                        |
| 4      | 80867.5                        | 80867.5                        |
| 5      | 80868.5                        | 80868.5                        |
| 6      | 80869.5                        | 80869.5                        |
| 7      | 80870.5                        | 80870.5                        |
| 8      | 80871.5                        | 80871.5                        |

```
# Get both stream time-series and time-stamps in a single data frame.
df = xdf.data()     # Accepts optional stream IDs
```

| sample | (1, &rsquo;ch:0&rsquo;) | (1, &rsquo;ch:1&rsquo;) | (1, &rsquo;time\_stamp&rsquo;) | (2, &rsquo;ch:0&rsquo;) | (2, &rsquo;ch:1&rsquo;) | (2, &rsquo;time\_stamp&rsquo;) |
|--------|-------------------------|-------------------------|--------------------------------|-------------------------|-------------------------|--------------------------------|
| 0      | 2                       | 2                       | 80863.5                        | 2                       | 2                       | 80863.5                        |
| 1      | 3                       | 3                       | 80864.5                        | 3                       | 3                       | 80864.5                        |
| 2      | 4                       | 4                       | 80865.5                        | 4                       | 4                       | 80865.5                        |
| 3      | 5                       | 5                       | 80866.5                        | 5                       | 5                       | 80866.5                        |
| 4      | 6                       | 6                       | 80867.5                        | 6                       | 6                       | 80867.5                        |
| 5      | 7                       | 7                       | 80868.5                        | 7                       | 7                       | 80868.5                        |
| 6      | 8                       | 8                       | 80869.5                        | 8                       | 8                       | 80869.5                        |
| 7      | 9                       | 9                       | 80870.5                        | 9                       | 9                       | 80870.5                        |
| 8      | 5                       | 5                       | 80871.5                        | 5                       | 5                       | 80871.5                        |

### Stream data MNE `RawArray`s

```
# Return stream data as an MNE RawArray.
raw = xdf.raw_mne(1)     # Accepts optional stream IDs
```

```
: Creating RawArray with float64 data, n_channels=2, n_times=9
:     Range : 0 ... 8 =      0.000 ...     8.000 secs
: Ready.
: []
```

```
# Multiple streams are returned as a list of RawArrays.
raws = xdf.raw_mne()     # Default return all loaded streams.
```

```
: Creating RawArray with float64 data, n_channels=2, n_times=9
:     Range : 0 ... 8 =      0.000 ...     8.000 secs
: Ready.
: []
: Creating RawArray with float64 data, n_channels=2, n_times=9
:     Range : 0 ... 8 =      0.000 ...     8.000 secs
: Ready.
: []
```
