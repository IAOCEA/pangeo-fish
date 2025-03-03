# Biologging Data Specification

The document describes the file structure and data format expected by `pangeo-fish` for the biologging (or "tag") data.
Indeed, tagging campaigns may involve different sensors, whose raw data are likely to not be dumped in a unified format.
To account for this diversity, we have defined a standardized format that we describe below.
We invite the user to follow along so that you can then translate your data to it.

## Overall Structure

The data repository contains one `.csv` file detailing all the acoustic receivers involved (named `stations.csv`), and each data storage tag (DST) data is stored under a directory whose name corresponds to the tag's identification:

```
your_tag_folder
├── A19124
│   ├── acoustic.csv
│   ├── dst.csv
│   ├── tagging_events.csv
│   └── metadata.json
├── ...
└── stations.csv
```

As illustrated above, each tag directory must contain the DST logs (`dst.csv`) and the lifetime data (`tagging_events.csv`).
It may also contain a file with all the acoustic detections that might have occurred (`acoustic.csv`).
Additional metadata can be specified in a `.json` file (`metadata.json`).

## CSV formatting

All CSV files must separate columns using `,` and new lines using `\n` characters.

All floating point (or fixed-point) values must use `.` as a decimal separator.

All time values must be given in ISO8601 format (`YYYY-MM-DDTHH:MM:SS±ZZZZ`, `YYYY-MM-DDTHH:MM:SSZ` or `YYYY-MM-DDTHH:MM:SS`).
Furthermore, time zone-naive datetime data (i.e. without timezone information) **will be assumed to be UTC**.

Strings containing `,` or `\n` must be wrapped in double quotes (`""`).

## `stations.csv`: information of the acoustic receivers

This file aims to encode the deployment information of the acoustic receivers.

If there are no acoustic detections, it may be removed.

It must contain at least eight columns:

- `deployment_id` and `station_name` describe the receivers' id and name, respectively.
- Deployment information in `deploy_time`, `deploy_longitude`, `deploy_latitude`.
- Similarly, recovry information in `recover_time`, `recover_longitude`, and `recover_latitude`.

It may contain arbitrary additional columns.

If the recover position is unknown or the same as the deployment position, it may be set to `NA`.

For example:

```
deployment_id,station_name,deploy_time,deploy_longitude,deploy_latitude,recover_time,recover_longitude,recover_latitude
12345,station_1,2022-06-13T06:14:00Z,-4.12345,49.56789,2022-06-20T14:07:00Z,NA,NA
12348,station_2,2022-06-13T05:51:00Z,-3.34567,46.23456,2022-06-20T14:00:00Z,NA,NA
```

## Inside a tag folder

### `dst.csv`: DST logs

The DST logs contains the data has been measured by the DST tag.

Current, `pangeo-fish` only supports temperature and pressure measurements.
A such, the file must have three columns: `time`, `temperature`, and `pressure` (and each row encodes a data point).

For example:

```
time,temperature,pressure
2022-07-21T12:12:30Z,10.1,14.3
2022-08-01T07:08:01Z,1.3,17.3
```

### `tagging_events.csv`: lifetime data

The lifetime data describes the start and end of the natural behavior of the fish, beginning with the release after the tagging and ending with its death/recapture.

It must have four columns: `event_name`, `time`, `latitude`, and `longitude`.

The valid values for `event_name` are: `release`, `recapture` and `fish_death` (more are possible but will be ignored).

The file must contain two entries: `release` and one of `recapture` or `fish_death`.

`latitude` and `longitude` are fixed-point representations of the position in degree, with a `.` as separator.
`latitude` must be in a range of `-90°` to `90°`, while `longitude` may be given in either `0°` to `360°` or `-180°` to `180°`.
If the position is unknown, both `latitude` and `longitude` must be set to `NA`.

For example:

```
event_name,time,latitude,longitude
release,2023-07-13T13:21:57Z,48.21842,-4.08578
recapture,2023-09-17T05:21:07Z,47.37423,-3.87582
```

or

```
event_name,time,latitude,longitude
release,2023-07-13T13:21:57Z,48.21842,-4.08578
fish_death,2023-09-17T05:21:07Z,NA,NA
```

### `acoustic.csv`: acoustic detections

This file contains information about acoustic detections.
If there are no acoustic detections, it may either be removed entirely or just contain the header (names of the columns).

It must contain at least two columns: `deployment_id` and `time`.

It may contain additional columns, such as the position of detection.

For example:

```
time,deployment_id
2022-08-10T22:11:00Z,176492
```

or

```
time,deployment_id,longitude,latitude
2022-08-10T22:11:00Z,176492,-3.88402,47.78820
2022-08-10T23:25:31Z,176492,-3.68471,47.81740
```

### `metadata.json`: arbitrary metadata

This file lets you add optional metadata.

It must be in JSON format and the top-level structure must be an object.
Additionally, the keys must be strings. Any valid JSON value is however allowed.

For example:

```json
{
  "pit_tag_id": "A19124",
  "acoustic_tag_id": "OPI-372",
  "fish_common_name": "sea bass",
  "tagging_campaign": "my_tagging_project"
}
```
