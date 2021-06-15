import torch
from denoising.models import ConvAutoencoder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from denoising.utils import apply_gaussian, create_coords
from denoising.data import CAEDataset
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import GSHHSFeature



def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def discretize(target, output):
    disc_output = output.copy()
    disc_target = target.copy()
    dmax = np.max(disc_output)
    dmin = np.min(disc_output)
    disc_output[disc_output < (dmax - dmin)/3] = dmin
    disc_output[np.logical_and(disc_output > (dmax - dmin)/3, disc_output < (dmax - dmin)/3*2 )] = (dmax - dmin)/2
    disc_output[disc_output > (dmax - dmin)/3*2] = dmax

    disc_target[disc_target < (dmax - dmin)/3] = dmin
    disc_target[np.logical_and(disc_target > (dmax - dmin)/3, disc_target < (dmax - dmin)/3*2 )] = (dmax - dmin)/2
    disc_target[disc_target > (dmax - dmin)/3*2] = dmax

    # target_norm = (target - np.min(output))/(np.max(output) - np.min(output))
    residual = output[0,0] - target[0]
    disc_residual = residual.copy()
    dmax = np.max(residual)
    dmin = np.min(residual)
    # print(dmin, dmax)
    disc_residual[np.logical_and(disc_residual > (dmax - dmin)/4, disc_residual < (dmax - dmin)/4*3 )] = (dmax - dmin)/2
    disc_residual[disc_residual < (dmax - dmin)/4] = dmin
    disc_residual[disc_residual > (dmax - dmin)/4*3] = dmax


def get_region_coords():
    x_coords = []
    y_coords = []
    # For threshold 64 North and South
    y0 = (90 - 64) * 4
    y1 = (180 - 26) * 4
    x0 = 0
    x1 = 360 * 4
    y = y0
    for i in range((y1 - y0) // 128):
        y = i * 128 + y0
        for j in range((x1 - x0) // 128):
            x = j * 128
            x_coords.append(x)
            y_coords.append(y)
    return x_coords, y_coords


def plot_region(tensor, ax, lons, lats, extent, crs=ccrs.PlateCarree(), **plot_kwargs):
    if 'cmap' not in plot_kwargs:
        plot_kwargs['cmap']='turbo'
    x0, x1, y0, y1 = extent
    ax.pcolormesh(lons, lats, tensor, transform=crs, **plot_kwargs)
    ax.set_extent((x0, x1, y0, y1), crs=crs)
    # longitude[longitude>180] = longitude[longitude>180] - 360
    ax.set_xticks(np.linspace(x1, x0, 5), crs=crs)
    ax.set_yticks(np.linspace(y0, y1, 5), crs=crs)
    lat_formatter = LatitudeFormatter()
    lon_formatter = LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.add_feature(GSHHSFeature(scale='intermediate', facecolor='lightgrey', linewidth=0.2))


def get_spatial_params(x, y, w=128, h=128, resolution=0.25):
    coord_grid = create_coords(resolution)
    lons = coord_grid[0][x:x+w]
    lats = coord_grid[1][int(180/resolution)-y:int(180/resolution)-(y+h):-1]
    x0, x1 = x * resolution, (x + w) * resolution
    y0, y1 = 90 - y * resolution, 90 - (y + h) * resolution
    if x0 > 180:
        x0 -= 360
        x1 -= 360
    extent = x0, x1, y0, y1
    return lons, lats, extent


class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


mdt = False
n_epochs = 200

if mdt:
    var = 'mdt'
else:
    var = 'cs'

model = ConvAutoencoder()
# model.load_state_dict(torch.load('./models/200e_mdt_model_autoencoder_unnorm.pth'))
model.load_state_dict(torch.load(f'./models/{n_epochs}e_{var}_model_cdae.pth'))
model.eval()

dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/{var}_testing_regions', quilt_dir=f'./quilting/DCGAN_{var}', mdt=mdt)

batch_size = 8
images = []
targets = []
paths = []
start = 29*13
for i in range(batch_size):
    image, target = dataset[i]
    images.append(image)
    targets.append(target)
    paths.append(dataset.paths[i])
images = torch.stack(images)
targets = torch.stack(targets)
print(images.size())

outputs = model(images)

outputs = outputs.detach().cpu().numpy()
targets = targets.detach().cpu().numpy()
images = images.detach().cpu().numpy()
mask_0 = targets == 0
mask = targets != 0

if not mdt:
    outputs = outputs * mask
    vmin = 0
    vmax = 2
else:
    vmin = -2
    vmax = 1

residual = (targets - outputs) * mask

# Plot input, target, outputs for multiple regions
crs = ccrs.PlateCarree()
fig, axes  = plt.subplots(3, batch_size, subplot_kw={'projection': crs})
if mdt:
    images[mask_0] = vmin
    outputs[mask_0] = vmin
    targets[mask_0] = vmin

for i, (axrow, tensor) in enumerate(zip(axes, [images, outputs, targets])):
    for j, ax in enumerate(axrow):
        split_path = paths[j][:len(paths[j])-4].split('_')
        x, y = int(split_path[-2]), int(split_path[-1])
        # w, h = tensor[j, 0].shape
        lons, lats, extent = get_spatial_params(x, y)
        plot_region(tensor[j, 0], ax, lons, lats, extent, vmin=vmin, vmax=vmax)
plt.show()
plt.close()


# Plot element wise RMSE across multiple regions
x_coords = [0, 128, 128, 256, 256, 384, 384, 512, 512, 768, 768, 768, 768, 896, 896, 896] # 640, 640, 640, 640
y_coords = [488, 360, 488, 360, 488, 360, 488, 232, 488, 104, 232, 360, 488, 232, 360, 488] # 104, 232, 360, 488
rmses = []
for x, y in zip(x_coords, y_coords):
    indices = []
    for i in range(len(dataset.paths)):
        split_path = dataset.paths[i][:len(dataset.paths[i])-4].split('_')
        a, b = int(split_path[-2]), int(split_path[-1])
        if x == a and y == b:
            indices.append(i)
    print(x, y)
    print(indices)
    regions = []
    target_regions = []
    for i in indices:
        region, target = dataset[i]
        regions.append(region)
        target_regions.append(target)
    regions = torch.stack(regions)
    target_regions = torch.stack(target_regions)
    output_regions = model(regions)

    target_regions = target_regions.detach().cpu().numpy()
    output_regions = output_regions.detach().cpu().numpy()
    mask = target_regions[0] != 0

    rmse = np.sqrt(np.mean((output_regions - target_regions)**2, axis=0))
    rmse = rmse * mask
    rmses.append(rmse[0])

fig, axes = plt.subplots(4, 4, subplot_kw={'projection': crs})
for i, axrow in enumerate(axes):
    for j, ax in enumerate(axrow):
        index = i * 4 + j
        x, y = x_coords[index], y_coords[index]
        lons, lats, extent = get_spatial_params(x, y)
        print(rmses[index].shape)
        plot_region(rmses[index], ax, lons, lats, extent,
            cmap='jet', vmin=0, vmax=0.35

        )
plt.show()





## regions = 0_488
# get all regions with same numbers
# save the indices of those regions

## indices = get_images_by_region
# use those indices to get items from the dataset 
## for i in indices:
##   image, target = dataset[i]
##   images.append(image)
##   targets.append(target)

# get batch of images on the same region

# compute predictions

# calculate squared pixel-wise error across batch axis with targets

# mean across batch axis then root





# # Plotting a signle input, output, target, residual
# ax[0].imshow(images, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[1].imshow(output, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[2].imshow(targets, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[3].imshow(residual, cmap='seismic', norm=MidpointNormalize(midpoint=0.))
# ax[0].title.set_text('Input')
# ax[1].title.set_text('Output')
# ax[2].title.set_text('Target')
# ax[3].title.set_text('Residual: (Output - Target)')
# plt.show()
# plt.close()

# # Plotting the signal
# fig, axes = plt.subplots(1, 3)
# axes[0].hist(image.ravel(), bins=20)
# axes[0].set_title('Input histogram')
# axes[1].hist(target.ravel(), bins=20)
# axes[1].set_title('Target histogram')
# axes[2].hist(output.ravel(), bins=20)
# axes[2].set_title('Output histogram')
# plt.show()

# # Scatter Diagram CorrCoef
# plt.plot(image.ravel(), target.ravel(), '.')
# plt.xlabel('Input signal')
# plt.ylabel('Target signal')
# plt.title('Input vs Target signal')
# np.corrcoef(image.ravel(), target.ravel())[0, 1]
# plt.show()
# plt.close()

# Plotting line of best fit
# m, b = np.polyfit(image, target, 1)
# plt.plot(image, m*image + b)
# plt.show()

# # 2D Histograms
# hist_2d, x_edges, y_edges = np.histogram2d(
#     image.ravel(),
#     target.ravel(),
#     bins=20)
# plt.imshow(hist_2d.T, origin='lower')
# plt.xlabel('Input')
# plt.ylabel('Target')
# plt.show()
# plt.close()

# # Log 2D Histograms
# hist_2d_log = np.zeros(hist_2d.shape)
# non_zeros = hist_2d != 0
# hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
# plt.imshow(hist_2d_log.T, origin='lower')
# plt.xlabel('Input')
# plt.ylabel('Target')
# plt.show()
# plt.close()


# # Compute Mutual Information
# print(mutual_information(hist_2d))

# # Discretize Plotting
# fig, ax  = plt.subplots(2,3)
# ax[0][0].imshow(image[0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[0][1].imshow(output[0,0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[0][2].imshow(target[0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[1][0].imshow(disc_output[0,0])
# ax[1][1].imshow(disc_target[0])
# ax[1][2].imshow(disc_output[0,0] - disc_target[0])
# ax[0][0].title.set_text('Input')
# ax[0][1].title.set_text('Output')
# ax[0][2].title.set_text('Target')
# ax[1][0].title.set_text('Discretized Output')
# ax[1][1].title.set_text('Discretized Target')
# ax[1][2].title.set_text('Disc. Output - Disc. Target') 
# plt.show()


# # Loading single regions for Gaussian filter plots
# target_img = np.load('../a_mdt_data/HR_model_data/training_regions/CNRM-CM6-1-HR_r1i1p1f2_gn_2001_cs_0_488.npy')
# quilt = Image.open('quilting/DCGAN_32deg/cut-b32-n5_2.png')
# quilt = quilt.convert(mode='L')
# quilt = np.array(quilt).astype(np.float32)
# mask = target_img != 0
# quilt = (quilt - np.nanmin(quilt)) / (np.nanmax(quilt) - np.nanmin(quilt))
# img = target_img + .3* quilt * mask

# # Gaussian
# mdt_img = np.load('../a_mdt_data/HR_model_data/training_regions/CNRM-CM6-1-HR_r1i1p1f2_gn_2001_cs_0_488.npy')
# gauss_img = apply_gaussian(img) * mask

# img = torch.tensor(img)
# print(img.shape)
# img = torch.unsqueeze(img, 0)
# img = torch.unsqueeze(img, 0)
# print(img.shape)

# # batch x channel x width x height
# output = model(img)
# # output = output.view(1, 1, 128, 128)
# output = output.detach().cpu().numpy()
# output = np.squeeze(output, 0)
# output = np.squeeze(output, 0)
# output = output * mask




