def center_longitude(ds):
    centered = (ds.longitude + 180) % 360 - 180
    return ds.assign_coords(longitude=centered)
