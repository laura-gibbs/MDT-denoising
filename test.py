import numpy as np
from PIL import Image

mdt = np.load('../a_mdt_data/HR_model_data/mdt_training_regions/CMCC-CM2-HR4_r1i1p1f1_gn_1851_0_488.npy')
cs = np.load('../a_mdt_data/HR_model_data/cs_training_regions/CMCC-CM2-HR4_r1i1p1f1_gn_1991_cs_256_360.npy')
new_cs = np.load('../a_mdt_data/HR_model_data/test/CMCC-CM2-HR4_r1i1p1f1_gn_1851_cs_128_488.npy')
quilt = Image.open('../MDT-Denoising/quilting/DCGAN_mdts/cut-b32-n5_0.png')

print(np.any(np.isnan(mdt)))
print(np.any(np.isnan(cs)))
print(np.any(np.isnan(new_cs)))
print(np.any(np.isnan(quilt)))
