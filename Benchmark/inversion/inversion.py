import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# --- RMSE calculation
"""

wasm_img = nib.load('./Benchmark/inversion/wasm_pipeline.nii')
reference_img = nib.load('./Benchmark/inversion/reference.nii')

wasm = wasm_img.get_fdata()
reference = reference_img.get_fdata()

mask = nib.load('./reference/mask.nii').get_fdata().astype(bool)


#there is an offset, so we have to normalize the data first:

#offset = np.mean(unwrapped[mask] - reference[mask])
#unwrapped = unwrapped - offset



difference = wasm[mask] - reference[mask]

squared_errors = difference ** 2
sum_squared_errors = np.sum(squared_errors)
n = np.sum(mask)
mean_squared_error = sum_squared_errors / n
rmse = np.sqrt(mean_squared_error)


print(f"RMSE: {rmse:.4f}")

range_ref = reference[mask].max() - reference[mask].min()
print(f"Range reference: {range_ref:.4f}")

range_ref = wasm[mask].max() - wasm[mask].min()
print(f"Range pipeline: {range_ref:.4f}")


"""





# --- Showing the voxel-wise residual error across the 3 axes for middle slices


wasm_img = nib.load('./Benchmark/inversion/wasm_pipeline.nii')
reference_img = nib.load('./Benchmark/inversion/reference.nii')

wasm = wasm_img.get_fdata()
reference = reference_img.get_fdata()

mask = nib.load('./reference/mask.nii').get_fdata().astype(bool)





voxelwise_residual = np.abs(wasm - reference)

# middle slice indexes
x, y, z = np.array(voxelwise_residual.shape) // 2


axial = np.where(mask[:, :, z], voxelwise_residual[:, :, z], np.nan)
coronal = np.where(mask[:, y, :], voxelwise_residual[:, y, :], np.nan)
sagittal = np.where(mask[x, :, :], voxelwise_residual[x, :, :], np.nan)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(axial.T, cmap='plasma', origin='lower')
axes[0].set_title(f'Axial (Z = {z})')

axes[1].imshow(coronal.T, cmap='plasma', origin='lower')
axes[1].set_title(f'Coronal (Y = {y})')

axes[2].imshow(sagittal.T, cmap='plasma', origin='lower')
axes[2].set_title(f'Sagittal (X = {x})')

for ax in axes:
    ax.axis('off')

range_ref = wasm[mask].max() - wasm[mask].min()
print("range wasm: ",range_ref)

range_ref = reference[mask].max() - reference[mask].min()
print("range reference: ",range_ref)
plt.tight_layout()
plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', label='Residual error')
plt.suptitle('Voxel-wise Residual Error', fontsize=14)
plt.show()

