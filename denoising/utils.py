


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((lat_range[0] + 90) * res, (lat_range[1] + 90) * res)

    return mdt[py[0]:py[1], px[0]:px[1]]