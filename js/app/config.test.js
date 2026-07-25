/**
 * Config Module Tests
 */
import {
  PHYSICS,
  VIEWER_CONFIG,
  MASK_CONFIG,
  BET_DEFAULTS,
  PIPELINE_DEFAULTS,
  TGV_DEFAULTS,
  QSMART_DEFAULTS,
  RTS_DEFAULTS,
  NDI_DEFAULTS,
  HDQSM_DEFAULTS,
  STAGE_DISPLAY_NAMES,
  getVoxelBasedDefaults
} from './config.js';

describe('Config Module', () => {
  describe('PHYSICS', () => {
    test('should have correct gyromagnetic ratio', () => {
      expect(PHYSICS.GYROMAGNETIC_RATIO).toBe(42.576e6);
    });
  });

  describe('VIEWER_CONFIG', () => {
    test('should have expected NiiVue settings', () => {
      expect(VIEWER_CONFIG.loadingText).toBe("");
      expect(VIEWER_CONFIG.dragToMeasure).toBe(false);
      expect(VIEWER_CONFIG.crosshairColor).toHaveLength(4);
      expect(VIEWER_CONFIG.crosshairWidth).toBe(0.75);
    });
  });

  describe('MASK_CONFIG', () => {
    test('should have sensible defaults', () => {
      expect(MASK_CONFIG.defaultThreshold).toBe(15);
      expect(MASK_CONFIG.defaultBrushSize).toBe(2);
    });
  });

  describe('BET_DEFAULTS', () => {
    test('should have expected values', () => {
      expect(BET_DEFAULTS.fractionalIntensity).toBe(0.5);
      expect(BET_DEFAULTS.iterations).toBe(1000);
      expect(BET_DEFAULTS.subdivisions).toBe(4);
    });
  });

  describe('TGV_DEFAULTS', () => {
    test('should have expected values', () => {
      expect(TGV_DEFAULTS.regularization).toBe(2);
      expect(TGV_DEFAULTS.iterations).toBe(1000);
      expect(TGV_DEFAULTS.erosions).toBe(3);
    });
  });

  describe('QSMART_DEFAULTS', () => {
    test('should have SDF parameters', () => {
      expect(QSMART_DEFAULTS.sdf_sigma1_stage1).toBe(10);
      expect(QSMART_DEFAULTS.sdf_lower_lim).toBe(0.6);
    });

    test('should have Frangi filter parameters in mm', () => {
      expect(QSMART_DEFAULTS.frangi_scale_min).toBe(0.5);
      expect(QSMART_DEFAULTS.frangi_scale_max).toBe(6.0);
    });

    test('should have iLSQR parameters', () => {
      expect(QSMART_DEFAULTS.ilsqr_tol).toBe(0.01);
      expect(QSMART_DEFAULTS.ilsqr_max_iter).toBe(50);
    });
  });

  describe('RTS_DEFAULTS', () => {
    test('should have expected values', () => {
      expect(RTS_DEFAULTS.delta).toBe(0.15);
      expect(RTS_DEFAULTS.mu).toBe(100000);
      expect(RTS_DEFAULTS.max_iter).toBe(20);
    });
  });

  describe('NDI_DEFAULTS', () => {
    test('should have expected values', () => {
      expect(NDI_DEFAULTS.tau).toBe(2);
      expect(NDI_DEFAULTS.alpha).toBe(0.00001);
      expect(NDI_DEFAULTS.max_iter).toBe(200);
    });
  });

  describe('HDQSM_DEFAULTS', () => {
    test('should have expected values', () => {
      expect(HDQSM_DEFAULTS.alpha_l2).toBe(0.0001);
      expect(HDQSM_DEFAULTS.mu1_l2).toBe(0.01);
      expect(HDQSM_DEFAULTS.max_iter_l1).toBe(20);
      expect(HDQSM_DEFAULTS.max_iter_l2).toBe(80);
    });
  });

  describe('STAGE_DISPLAY_NAMES', () => {
    test('should have all main stages', () => {
      expect(STAGE_DISPLAY_NAMES.magnitude).toBe('Magnitude');
      expect(STAGE_DISPLAY_NAMES.phase).toBe('Phase');
      expect(STAGE_DISPLAY_NAMES.mask).toBe('Mask');
      expect(STAGE_DISPLAY_NAMES.B0).toBe('B0 Field');
      expect(STAGE_DISPLAY_NAMES.final).toBe('QSM');
    });

    test('should have QSMART stages', () => {
      expect(STAGE_DISPLAY_NAMES.lfsStage1).toBe('LFS 1');
      expect(STAGE_DISPLAY_NAMES.chiStage1).toBe('χ1');
      expect(STAGE_DISPLAY_NAMES.vasculature).toBe('Vessels');
    });
  });

  describe('PIPELINE_DEFAULTS', () => {
    test('should have all algorithm settings', () => {
      expect(PIPELINE_DEFAULTS.combined_method).toBe('none');
      expect(PIPELINE_DEFAULTS.unwrapping_algorithm).toBe('romeo');
      expect(PIPELINE_DEFAULTS.bf_algorithm).toBe('vsharp');
      expect(PIPELINE_DEFAULTS.dipole_inversion).toBe('rts');
    });

    test('should have nested algorithm settings', () => {
      expect(PIPELINE_DEFAULTS.tgv).toBeDefined();
      expect(PIPELINE_DEFAULTS.qsmart).toBeDefined();
      expect(PIPELINE_DEFAULTS.vsharp).toBeDefined();
      expect(PIPELINE_DEFAULTS.rts).toBeDefined();
      expect(PIPELINE_DEFAULTS.medi).toBeDefined();
    });
  });

  describe('getVoxelBasedDefaults', () => {
    test('should calculate V-SHARP radii for isotropic voxels', () => {
      const defaults = getVoxelBasedDefaults([1, 1, 1]);

      // maxRadius = 18 * min(vsz) = 18
      expect(defaults.vsharpMaxRadius).toBe(18);
      // minRadius = max(1, 2 * min(vsz)) = 2
      expect(defaults.vsharpMinRadius).toBe(2);
    });

    test('should calculate SHARP radius', () => {
      const defaults = getVoxelBasedDefaults([1, 1, 1]);
      // radius = 18 * min(vsz) = 18
      expect(defaults.sharpRadius).toBe(18);
    });

    test('should calculate iSMV radius', () => {
      const defaults = getVoxelBasedDefaults([1, 1, 1]);
      // radius = max(2, 2 * max(vsz)) = 2
      expect(defaults.ismv_radius).toBe(2);
    });

    test('should handle anisotropic voxels', () => {
      const defaults = getVoxelBasedDefaults([0.5, 0.5, 2]);

      // min = 0.5, max = 2
      expect(defaults.vsharpMaxRadius).toBe(Math.round(18 * 0.5));  // 9
    });

    test('should calculate PDF maxit from mask dimensions', () => {
      const maskDims = [100, 100, 100];  // 1M voxels
      const defaults = getVoxelBasedDefaults([1, 1, 1], maskDims);

      // maxit = ceil(sqrt(1000000)) = 1000
      expect(defaults.pdfMaxit).toBe(1000);
    });

    test('should use default mask size when not provided', () => {
      const defaults = getVoxelBasedDefaults([1, 1, 1], null);

      // Default: 100000 -> sqrt = ~316
      expect(defaults.pdfMaxit).toBe(Math.ceil(Math.sqrt(100000)));
    });

    test('should use default voxel size when not provided', () => {
      const defaults = getVoxelBasedDefaults();

      // Default [1, 1, 1]
      expect(defaults.vsharpMaxRadius).toBe(18);
    });
  });
});
