import numpy as np
import nibabel as nib

def scale_phase_to_pi_range(phase_path, output_path):
    """
    Scale phase values from any range to [-π, +π] range.
    
    Parameters:
    phase_path (str): Path to input phase NIfTI file
    output_path (str): Path to save scaled phase NIfTI file
    
    Returns:
    str: Path to the scaled phase file
    """
    print(f"Loading phase data from: {phase_path}")
    
    # Load the phase image
    phase_img = nib.load(phase_path)
    phase_data = phase_img.get_fdata()
    
    # Get original data type info
    print(f"Original phase data shape: {phase_data.shape}")
    print(f"Original phase data type: {phase_data.dtype}")
    print(f"Original phase range: [{np.min(phase_data):.2f}, {np.max(phase_data):.2f}]")
    
    # Find the actual min and max values (ignoring any potential NaN values)
    phase_min = np.nanmin(phase_data)
    phase_max = np.nanmax(phase_data)
    
    print(f"Min phase value: {phase_min}")
    print(f"Max phase value: {phase_max}")
    
    # Check if already in correct range (with some tolerance)
    if abs(phase_min + np.pi) < 0.1 and abs(phase_max - np.pi) < 0.1:
        print("Phase data already appears to be in [-π, +π] range, copying as-is")
        scaled_phase = phase_data.astype(np.float32)
    else:
        # Scale to [0, 1] first
        if phase_max == phase_min:
            print("Warning: Phase data has zero range, setting to zero")
            scaled_phase = np.zeros_like(phase_data, dtype=np.float32)
        else:
            print("Scaling phase data to [-π, +π] range...")
            # Scale to [0, 1]
            normalized_phase = (phase_data - phase_min) / (phase_max - phase_min)
            
            # Scale to [-π, +π]
            scaled_phase = (normalized_phase * 2 * np.pi - np.pi).astype(np.float32)
    
    print(f"Scaled phase range: [{np.min(scaled_phase):.4f}, {np.max(scaled_phase):.4f}]")
    print(f"Expected range: [{-np.pi:.4f}, {np.pi:.4f}]")
    print(f"Scaled phase data type: {scaled_phase.dtype}")
    
    # Create new NIfTI image with scaled data, preserving header info
    scaled_img = nib.Nifti1Image(scaled_phase, phase_img.affine, phase_img.header)
    
    # Save the scaled phase
    nib.save(scaled_img, output_path)
    print(f"Scaled phase saved to: {output_path}")
    
    return output_path

def run_phase_scaling(phase_path):
    """
    Wrapper function for phase scaling that follows the naming convention
    of other processing functions.
    
    Parameters:
    phase_path (str): Path to input phase file
    
    Returns:
    str: Path to scaled phase file
    """
    import os
    
    # Create output path
    base_name = os.path.splitext(os.path.basename(phase_path))[0]
    output_path = f"{base_name}_scaled.nii"
    
    return scale_phase_to_pi_range(phase_path, output_path)

# Test function to verify scaling
def verify_phase_scaling(scaled_phase_path):
    """
    Verify that phase scaling was successful.
    """
    img = nib.load(scaled_phase_path)
    data = img.get_fdata()
    
    phase_min = np.nanmin(data)
    phase_max = np.nanmax(data)
    
    print(f"Verification - Phase range: [{phase_min:.4f}, {phase_max:.4f}]")
    print(f"Expected range: [{-np.pi:.4f}, {np.pi:.4f}]")
    
    # Check if in expected range (with small tolerance for floating point)
    tolerance = 0.01
    if (phase_min >= -np.pi - tolerance and phase_max <= np.pi + tolerance):
        print("✓ Phase scaling verification passed!")
        return True
    else:
        print("✗ Phase scaling verification failed!")
        return False