import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


#------
wrapped_img = nib.load('./phase.nii')
unwrapped_img = nib.load('./test_algo/unwrapped.nii')
wrapped_phase = wrapped_img.get_fdata()
unwrapped_phase = unwrapped_img.get_fdata()
#-----

# --- Compute phase residual

"""
mask = nib.load('./mask.nii').get_fdata().astype(bool)

wrapped_phase = wrapped_img.get_fdata()
unwrapped_phase = unwrapped_img.get_fdata()

#Compute phase residual
phase_residual = np.angle(np.exp(1j * wrapped_phase) * np.conj(np.exp(1j * unwrapped_phase)))
# Apply mask to phase residual
phase_residual = np.where(mask, phase_residual, np.nan)

# Plotting the phase residual 

# middle slice in z-dimension (randomly chosen)
z = phase_residual.shape[2] // 2

plt.imshow(phase_residual[:, :, z], cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.colorbar(label='Phase residual(radians)')
plt.title(f'Phase Residual between wrapped and unwrapped Slice z={z}')
plt.axis('off')
plt.show()
"""



# --- binary phase residual plot -> which voxel is correct and which not


"""


phase_residual = np.angle(np.exp(1j * wrapped_phase) * np.conj(np.exp(1j * unwrapped_phase)))

# Define a tolerance — anything within this range is considered "perfect"
tolerance = 1e-3  # radians

# Create binary mask: 1 if residual error is close to 0
perfect_unwrap_mask = np.abs(phase_residual) < tolerance

z = phase_residual.shape[2] // 2
plt.imshow(perfect_unwrap_mask[:, :, z], cmap='gray', vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0 = Error', '1 = Perfect'])
cbar.set_label('Unwrapping Accuracy')
plt.title(f'Unwrapping Accuracy Mask (Slice z={z})')
plt.axis('off')
plt.show()

"""

"""
# --- RMSE calculation

#unwrapped_img = nib.load('./test_algo/unwrapped.nii')
unwrapped_img = nib.load('./unwrapped_echo1.nii')
reference_img = nib.load('./reference/fieldmap.nii')

unwrapped = unwrapped_img.get_fdata()
reference = reference_img.get_fdata()

mask = nib.load('./mask.nii').get_fdata().astype(bool)


#there is an offset, so we have to normalize the data first:

offset = np.mean(unwrapped[mask] - reference[mask])
unwrapped = unwrapped - offset

# RMSE
difference = unwrapped[mask] - reference[mask]

print(unwrapped[mask].shape)

squared_errors = difference ** 2
sum_squared_errors = np.sum(squared_errors)
n = np.sum(mask)
mean_squared_error = sum_squared_errors / n
rmse = np.sqrt(mean_squared_error)


print(f"RMSE: {rmse:.4f} radians")

range_ref = reference[mask].max() - reference[mask].min()
nrmse_percent = (rmse / range_ref) * 100



print(f"RMSE percentage in reference range: {nrmse_percent:.4f} %")

# === Optional: Show histogram of errors ===
textstr = f"RMSE: {rmse:.4f} rad\nRMSE/(ref.max-ref.min): {nrmse_percent:.2f} %"
plt.hist(difference, bins=100, color='gray')
plt.title('Histogram of Voxel-wise Residual Error Uncentered (Unwrapped - Reference)')
plt.xlabel('Phase Error (radians)')
plt.ylabel('Voxel Count')
plt.grid(True)
plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

plt.show()
"""

# --- Showing the voxel-wise residual error across the 3 axes for middle slices


unwrapped_img = nib.load('./unwrapped_echo1.nii')
reference_img = nib.load('./reference/fieldmap.nii')
mask = nib.load('./mask.nii').get_fdata().astype(bool)

unwrapped = unwrapped_img.get_fdata()
reference = reference_img.get_fdata()


# offset correction
offset = np.mean(unwrapped[mask] - reference[mask])
unwrapped = unwrapped - offset


voxelwise_residual = np.abs(unwrapped - reference)

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

plt.tight_layout()
plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', label='Phase Error (radians)')
plt.suptitle('Voxel-wise Residual Error Centered (Middle Slices)', fontsize=14)
plt.show()

