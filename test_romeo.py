"""
ROMEO Python Implementation Tests
=================================

Unit tests for the pixel-perfect ROMEO implementation
Tests individual components and end-to-end functionality
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json
import time
from romeo_python import (
    romeo_unwrap, romeo_multi_echo_unwrap, calculateB0_unwrapped,
    unwrap_voxel, julia_compatible_mask
)


def test_unwrap_voxel():
    """Test the unwrap_voxel function"""
    print("Testing unwrap_voxel function...")
    
    # Test cases that should match Julia unwrapvoxel
    test_cases = [
        (1.0, 0.5, 1.0),          # Normal case
        (3.0, 0.5, 3.0),          # No wrap needed
        (-3.0, 0.5, 3.283),       # Negative wrap
        (0.1, 6.0, 6.383),        # Reference wrap
        (np.pi + 0.1, 0.1, 3.242), # π wrap
        (-np.pi - 0.1, 0.1, 3.042), # -π wrap
    ]
    
    passed = 0
    for new_val, old_val, expected in test_cases:
        result = unwrap_voxel(new_val, old_val)
        if abs(result - expected) < 0.001:
            passed += 1
        else:
            print(f"  FAIL: unwrap_voxel({new_val:.3f}, {old_val:.3f}) = {result:.3f}, expected {expected:.3f}")
    
    print(f"  unwrap_voxel: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_mask_generation():
    """Test Julia-compatible mask generation"""
    print("Testing mask generation...")
    
    # Create test magnitude data
    mag = np.random.uniform(20, 30, (16, 16, 8))
    
    # Test mask generation
    mask = julia_compatible_mask(mag)
    
    # Should be nearly 100% coverage (only excluding near-zero values)
    coverage = np.sum(mask) / mask.size * 100
    
    if coverage > 99.0:
        print(f"  Mask generation: PASS (coverage: {coverage:.1f}%)")
        return True
    else:
        print(f"  Mask generation: FAIL (coverage: {coverage:.1f}%, expected >99%)")
        return False


def test_small_region_unwrapping():
    """Test unwrapping on a small synthetic region"""
    print("Testing small region unwrapping...")
    
    # Create synthetic multi-echo data
    nx, ny, nz, necho = 16, 16, 8, 4
    TEs = np.array([4.0, 12.0, 20.0, 28.0])
    
    # Create synthetic B0 field
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    B0_true = 50 * np.sin(x/nx * 2*np.pi) * np.cos(y/ny * 2*np.pi)  # Hz
    
    # Generate wrapped phase for each echo
    phase = np.zeros((nx, ny, nz, necho))
    magnitude = np.ones((nx, ny, nz, necho)) * 100  # Uniform magnitude
    
    for i, te in enumerate(TEs):
        true_phase = B0_true * 2 * np.pi * te / 1000  # Convert TE to seconds
        phase[:, :, :, i] = np.angle(np.exp(1j * true_phase))  # Wrapped phase
    
    # Test individual unwrapping
    unwrapped = romeo_unwrap(phase, TEs, magnitude, individual=True)
    
    # Calculate B0 from unwrapped phase
    B0_calc = calculateB0_unwrapped(unwrapped, magnitude, TEs)
    
    # Check accuracy
    error = np.mean(np.abs(B0_calc - B0_true))
    
    if error < 5.0:  # 5 Hz tolerance
        print(f"  Small region unwrapping: PASS (B0 error: {error:.2f} Hz)")
        return True
    else:
        print(f"  Small region unwrapping: FAIL (B0 error: {error:.2f} Hz)")
        return False


def test_real_data_unwrapping(data_dir):
    """Test unwrapping on real MEGRE data"""
    print("Testing real data unwrapping...")
    
    try:
        # Load metadata for echo times
        json_files = sorted(Path(data_dir).glob("*part-mag_MEGRE.json"))
        if len(json_files) < 4:
            print(f"  Real data test: SKIP (insufficient data files)")
            return True
        
        TEs = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                meta = json.load(f)
                TEs.append(meta['EchoTime'] * 1000)
        
        TEs = np.array(TEs)
        print(f"  Echo times: {TEs} ms")
        
        # Load test region (64x64x32)
        mag_files = sorted(Path(data_dir).glob("*part-mag_MEGRE.nii"))
        phase_files = sorted(Path(data_dir).glob("*part-phase_MEGRE.nii"))
        
        if len(mag_files) < 4 or len(phase_files) < 4:
            print(f"  Real data test: SKIP (insufficient NIfTI files)")
            return True
        
        # Load central region
        mag_img = nib.load(mag_files[0])
        full_shape = mag_img.shape
        
        cx, cy, cz = full_shape[0]//2, full_shape[1]//2, full_shape[2]//2
        size_xy, size_z = 64, 32
        
        x_start, x_end = cx - size_xy//2, cx + size_xy//2
        y_start, y_end = cy - size_xy//2, cy + size_xy//2
        z_start, z_end = cz - size_z//2, cz + size_z//2
        
        necho = len(TEs)
        magnitude = np.zeros((size_xy, size_xy, size_z, necho))
        phase = np.zeros((size_xy, size_xy, size_z, necho))
        
        for i in range(necho):
            mag_data = nib.load(mag_files[i]).get_fdata()
            phase_data = nib.load(phase_files[i]).get_fdata()
            
            # Scale phase to [-π, π] if needed
            if np.max(phase_data) > 10:
                phase_data = (phase_data / np.max(phase_data)) * 2 * np.pi - np.pi
            
            magnitude[:, :, :, i] = mag_data[x_start:x_end, y_start:y_end, z_start:z_end]
            phase[:, :, :, i] = phase_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        print(f"  Test region shape: {magnitude.shape}")
        
        # Test individual unwrapping
        start_time = time.time()
        results = romeo_multi_echo_unwrap(
            phase, magnitude, TEs,
            individual=True,
            B0_calculation=True
        )
        unwrap_time = time.time() - start_time
        
        unwrapped = results['unwrapped']
        B0 = results['B0']
        mask = julia_compatible_mask(magnitude[:, :, :, 0])
        
        # Check temporal consistency (key metric for validation)
        errors = []
        for echo in range(1, necho):
            expected = unwrapped[:, :, :, 0] * (TEs[echo] / TEs[0])
            actual = unwrapped[:, :, :, echo]
            
            diff = np.abs(expected[mask] - actual[mask])
            mean_error = np.mean(diff)
            errors.append(mean_error)
        
        avg_error = np.mean(errors)
        
        # Success criteria: temporal error < 2.0 rad (Julia reference is ~1.7 rad)
        if avg_error < 2.0:
            print(f"  Real data unwrapping: PASS")
            print(f"    Temporal error: {avg_error:.3f} rad (Julia ref: ~1.7 rad)")
            print(f"    Processing time: {unwrap_time:.1f} seconds")
            print(f"    B0 range: [{np.min(B0[mask]):.1f}, {np.max(B0[mask]):.1f}] Hz")
            return True
        else:
            print(f"  Real data unwrapping: FAIL (temporal error: {avg_error:.3f} rad)")
            return False
            
    except Exception as e:
        print(f"  Real data test: ERROR ({str(e)})")
        return False


def test_temporal_vs_individual():
    """Compare temporal and individual unwrapping modes"""
    print("Testing temporal vs individual unwrapping...")
    
    # Create simple test data
    nx, ny, nz, necho = 32, 32, 16, 4
    TEs = np.array([4.0, 12.0, 20.0, 28.0])
    
    # Create linear B0 gradient
    x = np.arange(nx).reshape(-1, 1, 1)
    B0_field = (x - nx/2) * 2  # Linear gradient in Hz
    
    # Generate phase data
    phase = np.zeros((nx, ny, nz, necho))
    magnitude = np.ones((nx, ny, nz, necho)) * 50
    
    for i, te in enumerate(TEs):
        true_phase = B0_field * 2 * np.pi * te / 1000
        phase[:, :, :, i] = np.angle(np.exp(1j * true_phase))
    
    # Test both modes
    unwrapped_individual = romeo_unwrap(phase, TEs, magnitude, individual=True)
    unwrapped_temporal = romeo_unwrap(phase, TEs, magnitude, individual=False, template=1)
    
    # Both should work reasonably well on this simple case
    B0_individual = calculateB0_unwrapped(unwrapped_individual, magnitude, TEs)
    B0_temporal = calculateB0_unwrapped(unwrapped_temporal, magnitude, TEs)
    
    error_individual = np.mean(np.abs(B0_individual - B0_field))
    error_temporal = np.mean(np.abs(B0_temporal - B0_field))
    
    if error_individual < 10 and error_temporal < 10:
        print(f"  Temporal vs individual: PASS")
        print(f"    Individual B0 error: {error_individual:.2f} Hz")
        print(f"    Temporal B0 error: {error_temporal:.2f} Hz")
        return True
    else:
        print(f"  Temporal vs individual: FAIL")
        print(f"    Individual B0 error: {error_individual:.2f} Hz")
        print(f"    Temporal B0 error: {error_temporal:.2f} Hz")
        return False


def run_all_tests(data_dir=None):
    """Run all ROMEO tests"""
    print("=" * 60)
    print("ROMEO Python Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("unwrap_voxel", test_unwrap_voxel),
        ("mask_generation", test_mask_generation),
        ("small_region_unwrapping", test_small_region_unwrapping),
        ("temporal_vs_individual", test_temporal_vs_individual),
    ]
    
    if data_dir:
        tests.append(("real_data_unwrapping", lambda: test_real_data_unwrapping(data_dir)))
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name.replace('_', ' ').title()}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎯 All tests passed! ROMEO implementation is ready for production.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    # Run tests with real data if available
    data_dir = "/home/ashley/repos/qsmci/datasets/challenges/bids/sub-1/anat/"
    
    if Path(data_dir).exists():
        print(f"Using real data from: {data_dir}")
        run_all_tests(data_dir)
    else:
        print("Real data not available, running basic tests only")
        run_all_tests()