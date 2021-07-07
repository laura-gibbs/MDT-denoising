from scipy.ndimage import gaussian_filter
import numpy as np
import math


def apply_gaussian(arr, sigma=3, bd=-1.5):
    V = arr.copy()
    mask = np.ones_like(arr)
    mask[np.isnan(arr)] = np.nan
    V[np.isnan(arr)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*arr.copy() + 1
    W[np.isnan(arr)] = 0
    WW = gaussian_filter(W, sigma=sigma)

    arr = VV/WW * mask
    # arr[np.isnan(arr)] = bd

    return arr


def rb_gaussian(II, JJ, lon_d, lat_d, data, mask, frad, fltr_type):
    r = 6378136.3
    sigma = frad/np.sqrt(2.0*np.log(2.0))

    # Convert lons/lats to radians
    lon = lon_d*math.pi/180.0
    lat = lat_d*math.pi/180.0

    # Filter
    # Should indices be 1 or 0?
    x1 = r*np.cos(lat[0])*np.cos(lon[0])
    y1 = r*np.cos(lat[0])*np.sin(lon[0])
    z1 = r*np.sin(lon[0])


    # Compute meridional radius of filter (assume same for all latitudes)
    ltrd = 0
    for ltrd in range():
        x2 = r*np.cos(lat[ltrd])*np.cos(lon[0])
        y2 = r*np.cos(lat[ltrd])*np.sin(lon[0])
        z2 = r*np.sin(lat[ltrd])
        sd = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        ang = 2.0*np.asin(0.5*sd/r)
        d = r*ang

        if fltr_type=='box' and d > frad:
            break
        elif fltr_type=='gsn' and d > 10.0*sigma:
            break
        elif fltr_type=='tgn' and d > 2*frad:
            break
        elif fltr_type=='han' and d > 2*frad:
            break
        elif fltr_type=='ham' and d > 2*frad:
            break
        # ltrd = ltrd + 1

    for j in range(JJ):
        print('working on row ', j)

        # Comput lat limits of filter window at lat j
        m1 = j - (ltrd - 1)
        if m1 < 1: 
            m1 = 1
        if m2 > JJ:
            m2 = JJ
        
        fltr = np.zeros(II, JJ)
        tfltr = np.zeros(II, JJ)

        # Should indices be 1 or 0?
        x1 = r*np.cos(lat[0])*np.cos(lon[0])
        y1 = r*np.cos(lat[0])*np.sin(lon[0])
        z1 = r*np.sin(lon[0])
        
        # Chosen to define them as follows because they weren't defined
        sdata = np.zeros(II, JJ)
        rsd = np.zeros(II,JJ)

        for m in [m1, m2]:
            for n in range(II):
                x2 = r*np.cos(lat[j])*np.cos(lon[0])
                y2 = r*np.cos(lat[j])*np.sin(lon[0])
                z2 = r*np.sin(lon[j])

                sd = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                ang = 2.0 * np.asin(0.5*sd/r)
                d = r * ang
                if fltr_type == 'box' and d <= frad:
                    fltr[n,m] = 1.0
                if fltr_type == 'gsn' and d <= 10.0 * sigma:
                    fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                if fltr_type == 'tgn' and d <= 2*frad:
                    fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                # Think the following may be the wrong way around
                if fltr_type == 'han' and d <= 2*frad:
                    fltr[n,m] = 0.54 + 0.46*np.cos(d* math.pi/(2*frad))
                if fltr_type == 'ham' and d <= 2*frad:
                    fltr[n,m] = 0.5 + 0.5*np.cos(d*math.pi/(2*frad))
        
        for i in range(II):
            if mask[i,j]==0.0:

            #Translate the filter weights
                if i==0:
                    tfltr=fltr
                else:
                    tfltr[:i-1,:] = fltr[II-i+2:II-1,:]
                    tfltr[i:II-1,:] = fltr[:II-i+1,:]



                # Apply filter
                sm = 0.0
                for n in range(II):
                    for m in [m1, m2]:
                        if tfltr[n,m] !=0.0 and mask[n,m]==0.0:
                            sdata[i,j] = sdata[i,j] + tfltr[n,m]*data[n,m]
                            sm = sm+tfltr[n,m]

                if sm!=0.0:
                    sdata[i,j] = sdata[i,j]/sm
                
                rsd[i,j] = data[i,j]-sdata[i,j]

    return sdata, rsd

def main():
    pass


if __name__ == "__main__":
    main()
