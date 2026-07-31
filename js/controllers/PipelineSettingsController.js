/**
 * Pipeline Settings Controller
 *
 * Manages the pipeline settings modal UI - form population, visibility toggling,
 * reset to defaults, and reading form values.
 */

import {
  PIPELINE_DEFAULTS as D,
  TGV_DEFAULTS, SWI_DEFAULTS, QSMART_DEFAULTS, MCPC3DS_DEFAULTS,
  ROMEO_DEFAULTS, LINEAR_FIT_DEFAULTS,
  VSHARP_DEFAULTS, SHARP_DEFAULTS, RESHARP_DEFAULTS, HARPERELLA_DEFAULTS,
  ISMV_DEFAULTS, PDF_DEFAULTS, LBV_DEFAULTS,
  TKD_DEFAULTS, TSVD_DEFAULTS, TIKHONOV_DEFAULTS,
  TV_DEFAULTS, RTS_DEFAULTS, NLTV_DEFAULTS, MEDI_DEFAULTS, TFI_DEFAULTS,
  NDI_DEFAULTS, FANSI_DEFAULTS, L1QSM_DEFAULTS, WHQSM_DEFAULTS, HDQSM_DEFAULTS,
} from '../app/config.js';

export class PipelineSettingsController {
  constructor(modalElement) {
    this.modal = modalElement;
    this.inputMode = 'dicom'; // 'dicom', 'raw', 'totalField', or 'localField'
    this._setupTabs();
    this._setupEventListeners();
  }

  /**
   * Set the current input mode - controls which pipeline sections are visible
   * @param {string} mode - 'raw', 'totalField', or 'localField'
   */
  setInputMode(mode) {
    this.inputMode = mode;
  }

  /**
   * Open the modal and populate form from settings
   * @param {Object} settings - Current pipeline settings
   * @param {Object} defaults - Voxel-based default values
   * @param {number} nEchoes - Number of echo files loaded
   * @param {boolean} hasMagnitude - Whether magnitude data is available
   */
  open(settings, defaults, nEchoes, hasMagnitude = true) {
    this.hasMagnitude = hasMagnitude;
    this.nEchoes = nEchoes;
    this._populateForm(settings, defaults);
    this.updateVisibility(nEchoes);
    this._switchTab('tabQsmPipeline');
    this.modal.classList.add('active');
  }

  /**
   * Close the modal
   */
  close() {
    this.modal.classList.remove('active');
  }

  /**
   * Reset form to default values
   * @param {Object} defaults - Voxel-based default values
   */
  reset(defaults) {
    // Combined method
    this._setEl('combined_method', D.combined_method);

    // TGV defaults
    this._setEl('tgvRegularization', 2); // UI preset level (qsmbly-specific)
    this._setEl('tgvIterations', TGV_DEFAULTS.iterations);
    this._setEl('tgvErosions', TGV_DEFAULTS.erosions);

    // TFI defaults
    this._setEl('tfiLambda', TFI_DEFAULTS.lambda);
    this._setEl('tfiPrecond', TFI_DEFAULTS.precond);

    // SWI defaults
    this._setEl('swiScaling', SWI_DEFAULTS.scaling);
    this._setEl('swiStrength', SWI_DEFAULTS.strength);
    this._setEl('swiHpSigmaX', SWI_DEFAULTS.hp_sigma[0]);
    this._setEl('swiHpSigmaY', SWI_DEFAULTS.hp_sigma[1]);
    this._setEl('swiHpSigmaZ', SWI_DEFAULTS.hp_sigma[2]);
    this._setEl('swiMipWindow', SWI_DEFAULTS.mip_window);

    // QSMART defaults
    this._setEl('qsmartSdfSigma1Stage1', QSMART_DEFAULTS.sdf_sigma1_stage1);
    this._setEl('qsmartSdfSigma2Stage1', QSMART_DEFAULTS.sdf_sigma2_stage1);
    this._setEl('qsmartSdfSigma1Stage2', QSMART_DEFAULTS.sdf_sigma1_stage2);
    this._setEl('qsmartSdfSigma2Stage2', QSMART_DEFAULTS.sdf_sigma2_stage2);
    this._setEl('qsmartSdfSpatialRadius', QSMART_DEFAULTS.sdf_spatial_radius);
    this._setEl('qsmartSdfLowerLim', QSMART_DEFAULTS.sdf_lower_lim);
    this._setEl('qsmartSdfCurvConstant', QSMART_DEFAULTS.sdf_curv_constant);
    this._setEl('qsmartVascSphereRadius', QSMART_DEFAULTS.vasc_sphere_radius);
    this._setEl('qsmartFrangiScaleMin', QSMART_DEFAULTS.frangi_scale_min);
    this._setEl('qsmartFrangiScaleMax', QSMART_DEFAULTS.frangi_scale_max);
    this._setEl('qsmartFrangiScaleRatio', QSMART_DEFAULTS.frangi_scale_ratio);
    this._setEl('qsmartFrangiC', QSMART_DEFAULTS.frangi_c);
    this._setEl('qsmartIlsqrTol', QSMART_DEFAULTS.ilsqr_tol);
    this._setEl('qsmartIlsqrMaxIter', QSMART_DEFAULTS.ilsqr_max_iter);
    this._setEl('qsmartInversionMethod', QSMART_DEFAULTS.inversion_algorithm);
    // QSMART inner inversion per-algorithm params (reuse standard algorithm defaults)
    this._setEl('qsmartTkdThreshold', TKD_DEFAULTS.threshold);
    this._setEl('qsmartTsvdThreshold', TSVD_DEFAULTS.threshold);
    this._setEl('qsmartTikhLambda', TIKHONOV_DEFAULTS.lambda);
    this._setEl('qsmartTikhReg', 'identity');
    this._setEl('qsmartTvLambda', TV_DEFAULTS.lambda);
    this._setEl('qsmartTvMaxIter', TV_DEFAULTS.max_iter);
    this._setEl('qsmartTvTol', TV_DEFAULTS.tol);
    this._setEl('qsmartRtsDelta', RTS_DEFAULTS.delta);
    this._setEl('qsmartRtsMu', RTS_DEFAULTS.mu);
    this._setEl('qsmartRtsRho', RTS_DEFAULTS.rho);
    this._setEl('qsmartRtsMaxIter', RTS_DEFAULTS.max_iter);
    this._setEl('qsmartNltvLambda', NLTV_DEFAULTS.lambda);
    this._setEl('qsmartNltvMu', NLTV_DEFAULTS.mu);
    this._setEl('qsmartNltvMaxIter', NLTV_DEFAULTS.max_iter);
    this._setEl('qsmartNltvTol', NLTV_DEFAULTS.tol);
    this._setEl('qsmartNltvNewtonMaxIter', NLTV_DEFAULTS.newton_max_iter);
    this._setEl('qsmartMediLambda', MEDI_DEFAULTS.lambda);
    this._setEl('qsmartMediPercentage', MEDI_DEFAULTS.percentage);
    this._setEl('qsmartMediMaxIter', MEDI_DEFAULTS.max_iter);
    this._setEl('qsmartMediCgMaxIter', MEDI_DEFAULTS.cg_max_iter);
    this._setChecked('qsmartMediSmv', MEDI_DEFAULTS.smv);
    this._setEl('qsmartMediSmvRadius', MEDI_DEFAULTS.smv_radius);
    this._setChecked('qsmartMediMerit', MEDI_DEFAULTS.merit);
    this._updateQsmartInnerVisibility();

    // Phase offset
    this._setChecked('phase_offset_enabled', true);
    this._setEl('phase_offset_method', D.phase_offset_method);
    this._setEl('mcpc3dsSigmaX', MCPC3DS_DEFAULTS.sigma[0]);
    this._setEl('mcpc3dsSigmaY', MCPC3DS_DEFAULTS.sigma[1]);
    this._setEl('mcpc3dsSigmaZ', MCPC3DS_DEFAULTS.sigma[2]);

    // Bipolar correction
    this._setChecked('bipolar_correctionEnabled', false);

    // Unwrap method
    this._setEl('unwrapping_algorithm', D.unwrapping_algorithm);
    const resetHint = document.getElementById('unwrapLockedHint');
    if (resetHint) resetHint.style.display = 'none';
    this._showEl('romeo_settings', true);
    this._showEl('laplacian_settings', false);

    // ROMEO weight checkboxes
    this._setChecked('romeo_phase_gradient_coherence', ROMEO_DEFAULTS.phase_gradient_coherence);
    this._setChecked('romeo_mag_coherence', ROMEO_DEFAULTS.mag_coherence);
    this._setChecked('romeo_mag_weight', ROMEO_DEFAULTS.mag_weight);
    this._setEl('romeo_multi_echo_mode', 'individual');
    this._setChecked('romeo_correct_global', true);
    this._setEl('romeo_template_echo', '1');

    // Field calculation method
    this._setEl('b0_estimation', D.b0_estimation);
    this._showEl('weighted_avg_settings', true);
    this._showEl('linear_fit_settings', false);
    this._setEl('b0_weight_type', D.b0_weight_type);

    // Linear fit defaults
    this._setChecked('linear_fit_estimate_offset', LINEAR_FIT_DEFAULTS.estimate_offset);

    // Background removal
    this._setEl('bf_algorithm', D.bf_algorithm);
    this._showEl('vsharp_settings', true);
    this._showEl('sharp_settings', false);
    this._showEl('resharp_settings', false);
    this._showEl('ismv_settings', false);
    this._showEl('pdf_settings', false);
    this._showEl('lbv_settings', false);
    this._showEl('harperella_settings', false);
    this._showEl('iharperella_settings', false);

    this._setEl('vsharpMaxRadius', defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', VSHARP_DEFAULTS.threshold);
    this._setEl('sharpRadius', defaults.sharpRadius);
    this._setEl('sharpThreshold', SHARP_DEFAULTS.threshold);
    this._setEl('ismv_radius', defaults.ismv_radius);
    this._setEl('ismvTol', ISMV_DEFAULTS.tol);
    this._setEl('ismvMaxit', ISMV_DEFAULTS.maxit);
    this._setEl('pdfTol', PDF_DEFAULTS.tol);
    this._setEl('pdfMaxit', defaults.pdfMaxit);
    this._setEl('lbvTol', LBV_DEFAULTS.tol);
    this._setEl('lbvMaxit', defaults.lbvMaxit);
    this._setEl('resharpRadius', RESHARP_DEFAULTS.radius);
    this._setEl('resharpTikReg', RESHARP_DEFAULTS.tik_reg);
    this._setEl('resharpTol', RESHARP_DEFAULTS.tol);
    this._setEl('resharpMaxIter', RESHARP_DEFAULTS.max_iter);
    this._setEl('harperellaRadius', HARPERELLA_DEFAULTS.radius);
    this._setEl('harperellaMaxIter', HARPERELLA_DEFAULTS.max_iter);
    this._setEl('iharperellaRadius', HARPERELLA_DEFAULTS.radius);
    this._setEl('iharperellaMaxIter', HARPERELLA_DEFAULTS.max_iter);

    // Dipole inversion
    this._setEl('dipole_method', D.dipole_inversion);
    this._showEl('tkd_settings', false);
    this._showEl('tsvd_settings', false);
    this._showEl('tikhonov_settings', false);
    this._showEl('tv_settings', false);
    this._showEl('rts_settings', true);
    this._showEl('nltv_settings', false);
    this._showEl('medi_settings', false);
    this._showEl('ndi_settings', false);
    this._showEl('fansi_settings', false);
    this._showEl('fansitgv_settings', false);
    this._showEl('l1qsm_settings', false);
    this._showEl('whqsm_settings', false);
    this._showEl('hdqsm_settings', false);
    this._showEl('ilsqr_settings', false);

    this._setEl('tkdThreshold', TKD_DEFAULTS.threshold);
    this._setEl('tsvdThreshold', TSVD_DEFAULTS.threshold);
    this._setEl('tikhLambda', TIKHONOV_DEFAULTS.lambda);
    this._setEl('tikhReg', 'identity');

    this._setEl('tvLambda', TV_DEFAULTS.lambda);
    this._setEl('tvMaxIter', TV_DEFAULTS.max_iter);
    this._setEl('tvTol', TV_DEFAULTS.tol);

    this._setEl('rtsDelta', RTS_DEFAULTS.delta);
    this._setEl('rtsMu', RTS_DEFAULTS.mu);
    this._setEl('rtsRho', RTS_DEFAULTS.rho);
    this._setEl('rtsMaxIter', RTS_DEFAULTS.max_iter);

    this._setEl('nltvLambda', NLTV_DEFAULTS.lambda);
    this._setEl('nltvMu', NLTV_DEFAULTS.mu);
    this._setEl('nltvMaxIter', NLTV_DEFAULTS.max_iter);
    this._setEl('nltvTol', NLTV_DEFAULTS.tol);
    this._setEl('nltvNewtonMaxIter', NLTV_DEFAULTS.newton_max_iter);

    this._setEl('ndiTau', NDI_DEFAULTS.tau);
    this._setEl('ndiAlpha', NDI_DEFAULTS.alpha);
    this._setEl('ndiMaxIter', NDI_DEFAULTS.max_iter);

    this._setEl('fansiAlpha1', FANSI_DEFAULTS.alpha1);
    this._setEl('fansiMu1', FANSI_DEFAULTS.mu1);
    this._setEl('fansiMu2', FANSI_DEFAULTS.mu2);
    this._setEl('fansiAlpha0', FANSI_DEFAULTS.alpha0);
    this._setEl('fansiMu0', FANSI_DEFAULTS.mu0);
    this._setEl('fansiMaxIter', FANSI_DEFAULTS.max_iter);
    this._setEl('fansiTolUpdate', FANSI_DEFAULTS.tol_update);

    this._setEl('fansitgvAlpha1', FANSI_DEFAULTS.alpha1);
    this._setEl('fansitgvMu1', FANSI_DEFAULTS.mu1);
    this._setEl('fansitgvMu2', FANSI_DEFAULTS.mu2);
    this._setEl('fansitgvAlpha0', FANSI_DEFAULTS.alpha0);
    this._setEl('fansitgvMu0', FANSI_DEFAULTS.mu0);
    this._setEl('fansitgvMaxIter', FANSI_DEFAULTS.max_iter);
    this._setEl('fansitgvTolUpdate', FANSI_DEFAULTS.tol_update);

    this._setEl('l1qsmAlpha1', L1QSM_DEFAULTS.alpha1);
    this._setEl('l1qsmMu1', L1QSM_DEFAULTS.mu1);
    this._setEl('l1qsmMu2', L1QSM_DEFAULTS.mu2);
    this._setEl('l1qsmMu3', L1QSM_DEFAULTS.mu3);
    this._setEl('l1qsmLambda', L1QSM_DEFAULTS.lambda);
    this._setEl('l1qsmMaxIter', L1QSM_DEFAULTS.max_iter);
    this._setEl('l1qsmTolUpdate', L1QSM_DEFAULTS.tol_update);

    this._setEl('whqsmAlpha1', WHQSM_DEFAULTS.alpha1);
    this._setEl('whqsmMu1', WHQSM_DEFAULTS.mu1);
    this._setEl('whqsmMu2', WHQSM_DEFAULTS.mu2);
    this._setEl('whqsmBeta', WHQSM_DEFAULTS.beta);
    this._setEl('whqsmMuh', WHQSM_DEFAULTS.muh);
    this._setEl('whqsmMaxIter', WHQSM_DEFAULTS.max_iter);
    this._setEl('whqsmTolUpdate', WHQSM_DEFAULTS.tol_update);

    this._setEl('hdqsmAlphaL2', HDQSM_DEFAULTS.alpha_l2);
    this._setEl('hdqsmMu1L2', HDQSM_DEFAULTS.mu1_l2);
    this._setEl('hdqsmMu2', HDQSM_DEFAULTS.mu2);
    this._setEl('hdqsmMaxIterL1', HDQSM_DEFAULTS.max_iter_l1);
    this._setEl('hdqsmMaxIterL2', HDQSM_DEFAULTS.max_iter_l2);
    this._setEl('hdqsmTolUpdate', HDQSM_DEFAULTS.tol_update);

    this._setEl('mediLambda', MEDI_DEFAULTS.lambda);
    this._setEl('mediPercentage', MEDI_DEFAULTS.percentage);
    this._setEl('mediMaxIter', MEDI_DEFAULTS.max_iter);
    this._setEl('mediCgMaxIter', MEDI_DEFAULTS.cg_max_iter);
    this._setChecked('mediSmv', MEDI_DEFAULTS.smv);
    this._setEl('mediSmvRadius', MEDI_DEFAULTS.smv_radius);
    this._showEl('mediSmvRadiusGroup', MEDI_DEFAULTS.smv);
    this._setChecked('mediMerit', MEDI_DEFAULTS.merit);

    this._setEl('ilsqr_tol', QSMART_DEFAULTS.ilsqr_tol);
    this._setEl('ilsqr_max_iter', QSMART_DEFAULTS.ilsqr_max_iter);
  }

  /**
   * Read form values and return settings object
   * @param {number} nEchoes - Number of echo files loaded
   * @returns {Object} Pipeline settings object
   */
  save(nEchoes) {
    const isMultiEcho = nEchoes > 1;

    const unwrapping_algorithm = isMultiEcho
      ? this._getEl('unwrapping_algorithm')
      : this._getEl('single_echo_unwrapping_algorithm');
    const isLaplacian = unwrapping_algorithm === 'laplacian';

    // Phase offset: disabled for Laplacian (inherently removes offsets)
    const phaseOffsetEnabled = !isLaplacian && isMultiEcho && (this._getChecked('phase_offset_enabled') ?? true);
    const phase_offset_method = phaseOffsetEnabled ? (this._getEl('phase_offset_method') || 'mcpc3ds') : 'none';

    // ROMEO weight settings
    const romeoPhaseGradientCoherence = isMultiEcho
      ? this._getChecked('romeo_phase_gradient_coherence') ?? true
      : true;
    const romeoMagCoherence = this._getChecked('romeo_mag_coherence') ?? true;
    const romeoMagWeight = this._getChecked('romeo_mag_weight') ?? true;

    return {
      combined_method: this._getEl('combined_method'),
      reference_mean: this._getChecked('qsm_reference_mean') ?? true,
      swi: {
        hp_sigma: [
          parseFloat(this._getEl('swiHpSigmaX')),
          parseFloat(this._getEl('swiHpSigmaY')),
          parseFloat(this._getEl('swiHpSigmaZ'))
        ],
        scaling: this._getEl('swiScaling') || 'tanh',
        strength: parseFloat(this._getEl('swiStrength')),
        mip_window: parseInt(this._getEl('swiMipWindow'))
      },
      tgv: {
        regularization: parseInt(this._getEl('tgvRegularization')),
        iterations: parseInt(this._getEl('tgvIterations')),
        erosions: parseInt(this._getEl('tgvErosions'))
      },
      tfi: {
        ...TFI_DEFAULTS,
        lambda: parseFloat(this._getEl('tfiLambda')),
        precond: parseFloat(this._getEl('tfiPrecond'))
      },
      qsmart: {
        sdf_sigma1_stage1: parseFloat(this._getEl('qsmartSdfSigma1Stage1')),
        sdf_sigma2_stage1: parseFloat(this._getEl('qsmartSdfSigma2Stage1')),
        sdf_sigma1_stage2: parseFloat(this._getEl('qsmartSdfSigma1Stage2')),
        sdf_sigma2_stage2: parseFloat(this._getEl('qsmartSdfSigma2Stage2')),
        sdf_spatial_radius: parseInt(this._getEl('qsmartSdfSpatialRadius')),
        sdf_lower_lim: parseFloat(this._getEl('qsmartSdfLowerLim')),
        sdf_curv_constant: parseFloat(this._getEl('qsmartSdfCurvConstant')),
        vasc_sphere_radius: parseFloat(this._getEl('qsmartVascSphereRadius')),
        frangi_scale_min: parseFloat(this._getEl('qsmartFrangiScaleMin')),
        frangi_scale_max: parseFloat(this._getEl('qsmartFrangiScaleMax')),
        frangi_scale_ratio: parseFloat(this._getEl('qsmartFrangiScaleRatio')),
        frangi_c: parseFloat(this._getEl('qsmartFrangiC')),
        ilsqr_tol: parseFloat(this._getEl('qsmartIlsqrTol')),
        ilsqr_max_iter: parseInt(this._getEl('qsmartIlsqrMaxIter')),
        inversion_algorithm: this._getEl('qsmartInversionMethod') || 'ilsqr',
        // Per-algorithm params for the inner inversion (used when the matching algorithm is selected)
        tkd: { threshold: parseFloat(this._getEl('qsmartTkdThreshold')) },
        tsvd: { threshold: parseFloat(this._getEl('qsmartTsvdThreshold')) },
        tikhonov: { lambda: parseFloat(this._getEl('qsmartTikhLambda')), reg: this._getEl('qsmartTikhReg') },
        tv: {
          lambda: parseFloat(this._getEl('qsmartTvLambda')),
          max_iter: parseInt(this._getEl('qsmartTvMaxIter')),
          tol: parseFloat(this._getEl('qsmartTvTol'))
        },
        rts: {
          delta: parseFloat(this._getEl('qsmartRtsDelta')),
          mu: parseFloat(this._getEl('qsmartRtsMu')),
          rho: parseFloat(this._getEl('qsmartRtsRho')),
          max_iter: parseInt(this._getEl('qsmartRtsMaxIter'))
        },
        nltv: {
          lambda: parseFloat(this._getEl('qsmartNltvLambda')),
          mu: parseFloat(this._getEl('qsmartNltvMu')),
          max_iter: parseInt(this._getEl('qsmartNltvMaxIter')),
          tol: parseFloat(this._getEl('qsmartNltvTol')),
          newton_max_iter: parseInt(this._getEl('qsmartNltvNewtonMaxIter'))
        },
        medi: {
          lambda: parseFloat(this._getEl('qsmartMediLambda')),
          percentage: parseFloat(this._getEl('qsmartMediPercentage')),
          max_iter: parseInt(this._getEl('qsmartMediMaxIter')),
          cg_max_iter: parseInt(this._getEl('qsmartMediCgMaxIter')),
          smv: this._getChecked('qsmartMediSmv'),
          smv_radius: parseFloat(this._getEl('qsmartMediSmvRadius')),
          merit: this._getChecked('qsmartMediMerit')
        }
      },
      unwrapping_algorithm: unwrapping_algorithm,
      phase_offset_method: phase_offset_method,
      bipolar_correction: !isLaplacian && isMultiEcho && nEchoes >= 3 && (this._getChecked('bipolar_correctionEnabled') ?? false),
      b0_estimation: this._getEl('b0_estimation') || 'weighted_avg',
      mcpc3ds: {
        sigma: [
          parseInt(this._getEl('mcpc3dsSigmaX')),
          parseInt(this._getEl('mcpc3dsSigmaY')),
          parseInt(this._getEl('mcpc3dsSigmaZ'))
        ]
      },
      b0_weight_type: this._getEl('b0_weight_type') || 'phase_snr',
      linearFit: {
        estimate_offset: this._getChecked('linear_fit_estimate_offset') ?? true
      },
      romeo: {
        phase_gradient_coherence: romeoPhaseGradientCoherence,
        mag_coherence: romeoMagCoherence,
        mag_weight: romeoMagWeight,
        individual: (this._getEl('romeo_multi_echo_mode') || 'individual') === 'individual',
        correct_global: this._getChecked('romeo_correct_global') ?? true,
        template: parseInt(this._getEl('romeo_template_echo') || '1') - 1,
      },
      bf_algorithm: this._getEl('bf_algorithm'),
      vsharp: {
        max_radius: parseFloat(this._getEl('vsharpMaxRadius')),
        min_radius: parseFloat(this._getEl('vsharpMinRadius')),
        threshold: parseFloat(this._getEl('vsharpThreshold'))
      },
      sharp: {
        radius: parseFloat(this._getEl('sharpRadius')),
        threshold: parseFloat(this._getEl('sharpThreshold'))
      },
      ismv: {
        radius: parseFloat(this._getEl('ismv_radius')),
        tol: parseFloat(this._getEl('ismvTol')),
        maxit: parseInt(this._getEl('ismvMaxit'))
      },
      pdf: {
        tol: parseFloat(this._getEl('pdfTol')),
        maxit: parseInt(this._getEl('pdfMaxit'))
      },
      resharp: {
        radius: parseFloat(this._getEl('resharpRadius')),
        tik_reg: parseFloat(this._getEl('resharpTikReg')),
        tol: parseFloat(this._getEl('resharpTol')),
        max_iter: parseInt(this._getEl('resharpMaxIter'))
      },
      harperella: {
        radius: parseFloat(this._getEl('harperellaRadius')),
        max_iter: parseInt(this._getEl('harperellaMaxIter')),
        tol: 1e-6
      },
      iharperella: {
        radius: parseFloat(this._getEl('iharperellaRadius')),
        max_iter: parseInt(this._getEl('iharperellaMaxIter')),
        tol: 1e-6
      },
      lbv: {
        tol: parseFloat(this._getEl('lbvTol')),
        maxit: parseInt(this._getEl('lbvMaxit'))
      },
      dipole_inversion: this._getEl('dipole_method'),
      tkd: {
        threshold: parseFloat(this._getEl('tkdThreshold'))
      },
      tsvd: {
        threshold: parseFloat(this._getEl('tsvdThreshold'))
      },
      tikhonov: {
        lambda: parseFloat(this._getEl('tikhLambda')),
        reg: this._getEl('tikhReg')
      },
      tv: {
        lambda: parseFloat(this._getEl('tvLambda')),
        max_iter: parseInt(this._getEl('tvMaxIter')),
        tol: parseFloat(this._getEl('tvTol'))
      },
      rts: {
        delta: parseFloat(this._getEl('rtsDelta')),
        mu: parseFloat(this._getEl('rtsMu')),
        rho: parseFloat(this._getEl('rtsRho')),
        max_iter: parseInt(this._getEl('rtsMaxIter'))
      },
      nltv: {
        lambda: parseFloat(this._getEl('nltvLambda')),
        mu: parseFloat(this._getEl('nltvMu')),
        max_iter: parseInt(this._getEl('nltvMaxIter')),
        tol: parseFloat(this._getEl('nltvTol')),
        newton_max_iter: parseInt(this._getEl('nltvNewtonMaxIter'))
      },
      ndi: {
        tau: parseFloat(this._getEl('ndiTau')),
        alpha: parseFloat(this._getEl('ndiAlpha')),
        max_iter: parseInt(this._getEl('ndiMaxIter'))
      },
      fansi: {
        alpha1: parseFloat(this._getEl('fansiAlpha1')),
        mu1: parseFloat(this._getEl('fansiMu1')),
        mu2: parseFloat(this._getEl('fansiMu2')),
        alpha0: parseFloat(this._getEl('fansiAlpha0')),
        mu0: parseFloat(this._getEl('fansiMu0')),
        max_iter: parseInt(this._getEl('fansiMaxIter')),
        tol_update: parseFloat(this._getEl('fansiTolUpdate'))
      },
      fansitgv: {
        alpha1: parseFloat(this._getEl('fansitgvAlpha1')),
        mu1: parseFloat(this._getEl('fansitgvMu1')),
        mu2: parseFloat(this._getEl('fansitgvMu2')),
        alpha0: parseFloat(this._getEl('fansitgvAlpha0')),
        mu0: parseFloat(this._getEl('fansitgvMu0')),
        max_iter: parseInt(this._getEl('fansitgvMaxIter')),
        tol_update: parseFloat(this._getEl('fansitgvTolUpdate'))
      },
      l1qsm: {
        alpha1: parseFloat(this._getEl('l1qsmAlpha1')),
        mu1: parseFloat(this._getEl('l1qsmMu1')),
        mu2: parseFloat(this._getEl('l1qsmMu2')),
        mu3: parseFloat(this._getEl('l1qsmMu3')),
        lambda: parseFloat(this._getEl('l1qsmLambda')),
        max_iter: parseInt(this._getEl('l1qsmMaxIter')),
        tol_update: parseFloat(this._getEl('l1qsmTolUpdate'))
      },
      whqsm: {
        alpha1: parseFloat(this._getEl('whqsmAlpha1')),
        mu1: parseFloat(this._getEl('whqsmMu1')),
        mu2: parseFloat(this._getEl('whqsmMu2')),
        beta: parseFloat(this._getEl('whqsmBeta')),
        muh: parseFloat(this._getEl('whqsmMuh')),
        max_iter: parseInt(this._getEl('whqsmMaxIter')),
        tol_update: parseFloat(this._getEl('whqsmTolUpdate'))
      },
      hdqsm: {
        alpha_l2: parseFloat(this._getEl('hdqsmAlphaL2')),
        mu1_l2: parseFloat(this._getEl('hdqsmMu1L2')),
        mu2: parseFloat(this._getEl('hdqsmMu2')),
        max_iter_l1: parseInt(this._getEl('hdqsmMaxIterL1')),
        max_iter_l2: parseInt(this._getEl('hdqsmMaxIterL2')),
        tol_update: parseFloat(this._getEl('hdqsmTolUpdate'))
      },
      medi: {
        lambda: parseFloat(this._getEl('mediLambda')),
        percentage: parseFloat(this._getEl('mediPercentage')),
        max_iter: parseInt(this._getEl('mediMaxIter')),
        cg_max_iter: parseInt(this._getEl('mediCgMaxIter')),
        cg_tol: 0.01,
        tol: 0.1,
        smv: this._getChecked('mediSmv'),
        smv_radius: parseFloat(this._getEl('mediSmvRadius')),
        merit: this._getChecked('mediMerit'),
        data_weighting: 1
      },
      ilsqr: {
        tol: parseFloat(this._getEl('ilsqr_tol')),
        max_iter: parseInt(this._getEl('ilsqr_max_iter'))
      }
    };
  }

  /**
   * Update visibility of sections based on method selections and echo count
   * @param {number} nEchoes - Number of echo files loaded
   */
  updateVisibility(nEchoes) {
    const isRawMode = this.inputMode === 'raw' || this.inputMode === 'dicom';
    const isTotalFieldMode = this.inputMode === 'totalField';
    const isLocalFieldMode = this.inputMode === 'localField';
    const isFieldMapMode = isTotalFieldMode || isLocalFieldMode;

    const combined_method = this._getEl('combined_method');
    const phase_offset_method = this._getEl('phase_offset_method') || 'mcpc3ds';
    const isTgv = combined_method === 'tgv';
    const isQsmart = combined_method === 'qsmart';
    const isTfi = combined_method === 'tfi';
    const isCombined = isTgv || isQsmart || isTfi;
    const isMcpc3ds = phase_offset_method === 'mcpc3ds';
    const isMultiEcho = nEchoes > 1;

    // Combined method selector - available in all modes
    const combined_methodGroup = document.getElementById('combined_method')?.closest('.param-group');
    if (combined_methodGroup) combined_methodGroup.style.display = '';

    // TGV settings - show when TGV selected in any mode
    this._showEl('tgv_settings', isTgv);

    // QSMART settings - show when QSMART selected in any mode
    this._showEl('qsmart_settings', isQsmart);
    if (isQsmart) this._updateQsmartInnerVisibility();

    // TFI settings - show when TFI selected in any mode
    this._showEl('tfi_settings', isTfi);

    // Phase unwrapping (check this first — Laplacian disables offset removal + bipolar)
    const currentUnwrapMethod = this._getEl('unwrapping_algorithm') || 'romeo';
    const isLaplacian = currentUnwrapMethod === 'laplacian';
    this._showEl('romeo_settings', currentUnwrapMethod === 'romeo');
    this._showEl('laplacian_settings', isLaplacian);

    // ROMEO multi-echo mode (only shown for multi-echo + ROMEO)
    this._showEl('romeo_multi_echo_settings', currentUnwrapMethod === 'romeo' && isMultiEcho);

    // Phase offset removal — disabled for Laplacian (it inherently removes offsets)
    const phaseOffsetEnabled = !isLaplacian && (this._getChecked('phase_offset_enabled') ?? true);
    this._disableEl('phase_offset_content', !phaseOffsetEnabled);
    const phaseOffsetCheckbox = document.getElementById('phase_offset_enabled');
    if (phaseOffsetCheckbox) phaseOffsetCheckbox.disabled = isLaplacian;
    this._showWarning('phase_offset_enabled', 'phaseOffsetWarning',
      phaseOffsetEnabled && nEchoes === 1,
      'Requires multi-echo data', 'error');
    this._showWarning('phase_offset_enabled', 'phaseOffsetLaplacianNote',
      isLaplacian && isMultiEcho,
      'Not needed — Laplacian unwrapping inherently removes phase offsets',
      'info');


    // Bipolar correction — disabled for Laplacian
    const bipolarCheckbox = document.getElementById('bipolar_correctionEnabled');
    if (bipolarCheckbox) bipolarCheckbox.disabled = isLaplacian;
    const bipolarEnabled = !isLaplacian && (this._getChecked('bipolar_correctionEnabled') ?? false);
    this._showWarning('bipolar_correctionEnabled', 'bipolarWarning',
      bipolarEnabled && nEchoes >= 1 && nEchoes < 3,
      nEchoes === 1 ? 'Requires multi-echo data (3+ echoes)' : 'Requires 3+ echoes',
      'error');
    this._showWarning('bipolar_correctionEnabled', 'bipolarLaplacianNote',
      isLaplacian,
      'Not applicable with Laplacian unwrapping',
      'info');

    // Phase Gradient Coherence only meaningful for multi-echo
    const pgcLabel = document.getElementById('romeoPgcLabel');
    if (pgcLabel) pgcLabel.style.display = isMultiEcho ? '' : 'none';

    // Multi-echo fitting
    const fieldCalcMethod = this._getEl('b0_estimation') || 'weighted_avg';
    this._showEl('weighted_avg_settings', fieldCalcMethod === 'weighted_avg');
    this._showEl('linear_fit_settings', fieldCalcMethod === 'linear_fit');
    this._showWarning('b0_estimation', 'multiEchoFittingWarning',
      nEchoes === 1, 'Requires multi-echo data', 'error');

    // Background removal - show for:
    // - Raw mode standard pipeline (not TGV/QSMART)
    // - Total field standard pipeline
    // - Total field + QSMART (QSMART uses SDF for total field)
    // NOT shown for: TGV (handles BG removal internally), local field, raw+QSMART (handled internally)
    const showBgRemoval = (!isCombined && isRawMode) || (isTotalFieldMode && !isTgv && !isQsmart);
    this._showEl('bgRemovalSection', showBgRemoval);

    // Dipole inversion - show for standard pipeline only (TGV/QSMART handle inversion internally)
    const showDipoleInversion = (!isCombined && isRawMode) || (!isCombined && isFieldMapMode);
    this._showEl('dipole_inversionSection', showDipoleInversion);

    // Check if MEDI with SMV is enabled - show error on background removal
    const dipoleMethod = this._getEl('dipole_method');
    const mediSmvEnabled = this._getChecked('mediSmv');
    const bgDisabledByMediSmv = dipoleMethod === 'medi' && mediSmvEnabled && showBgRemoval;

    const bgHint = document.getElementById('bgRemovalDisabledHint');
    if (bgHint) bgHint.style.display = bgDisabledByMediSmv ? '' : 'none';

    // Enable/disable tabs based on pipeline state
    this._setTabEnabled('tabPhaseProcessing', isRawMode);
    this._setTabEnabled('tabBgRemoval', showBgRemoval);
    this._setTabEnabled('tabDipoleInversion', showDipoleInversion);

    // Show errors for magnitude-dependent features when no magnitude is available
    const noMag = this.hasMagnitude === false;

    // QSMART option in combined method dropdown
    const combinedSelect = document.getElementById('combined_method');
    if (combinedSelect) {
      this._showWarning('combined_method', 'combined_methodWarning',
        noMag && combinedSelect.value === 'qsmart',
        'Requires magnitude', 'error');
    }

    // MEDI option in dipole inversion dropdown
    const dipoleSelect = document.getElementById('dipole_method');
    if (dipoleSelect) {
      this._showWarning('dipole_method', 'dipoleMethodWarning',
        noMag && dipoleSelect.value === 'medi',
        'Requires magnitude', 'error');
    }

    // ROMEO magnitude weight checkboxes
    const romeoMagCoh = document.getElementById('romeo_mag_coherence');
    const romeoMagWt = document.getElementById('romeo_mag_weight');
    if (romeoMagCoh) {
      this._showWarning('romeo_mag_coherence', 'romeoMagCohWarning',
        noMag && romeoMagCoh.checked, 'Requires magnitude', 'error');
    }
    if (romeoMagWt) {
      this._showWarning('romeo_mag_weight', 'romeoMagWtWarning',
        noMag && romeoMagWt.checked, 'Requires magnitude', 'error');
    }

    // B0 weight type — phase_snr, phase_var, and mag all require magnitude
    const b0WeightSelect = document.getElementById('b0_weight_type');
    if (b0WeightSelect) {
      const mag_weights = ['phase_snr', 'phase_var', 'mag'];
      this._showWarning('b0_weight_type', 'b0WeightWarning',
        noMag && mag_weights.includes(b0WeightSelect.value),
        'Requires magnitude', 'error');
    }
  }

  // ---- Tab management ----

  _setupTabs() {
    const tabBar = this.modal.querySelector('.pipeline-tabs');
    if (!tabBar) return;
    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.pipeline-tab');
      if (!btn || btn.disabled) return;
      this._switchTab(btn.dataset.tab);
    });
  }

  _switchTab(tabId) {
    const tabBar = this.modal.querySelector('.pipeline-tabs');
    if (!tabBar) return;
    for (const btn of tabBar.querySelectorAll('.pipeline-tab')) {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    }
    const body = this.modal.querySelector('.modal-body');
    if (!body) return;
    for (const panel of body.querySelectorAll('.tab-panel')) {
      panel.classList.toggle('active', panel.id === tabId);
    }
  }

  _setTabEnabled(tabId, enabled) {
    const btn = this.modal.querySelector(`.pipeline-tab[data-tab="${tabId}"]`);
    if (!btn) return;
    btn.disabled = !enabled;
    // If the disabled tab was active, switch to first enabled tab
    if (!enabled && btn.classList.contains('active')) {
      const first = this.modal.querySelector('.pipeline-tab:not(:disabled)');
      if (first) this._switchTab(first.dataset.tab);
    }
  }

  // ---- Private helpers ----

  _populateForm(settings, defaults) {
    // Combined method
    this._setEl('combined_method', settings.combined_method || 'none');

    // SWI settings
    const swiSettings = settings.swi || {};
    this._setEl('swiScaling', swiSettings.scaling || 'tanh');
    this._setEl('swiStrength', swiSettings.strength ?? 4);
    this._setEl('swiHpSigmaX', swiSettings.hp_sigma?.[0] ?? 4);
    this._setEl('swiHpSigmaY', swiSettings.hp_sigma?.[1] ?? 4);
    this._setEl('swiHpSigmaZ', swiSettings.hp_sigma?.[2] ?? 0);
    this._setEl('swiMipWindow', swiSettings.mip_window ?? 7);

    // TGV settings
    this._setEl('tgvRegularization', settings.tgv.regularization);
    this._setEl('tgvIterations', settings.tgv.iterations);
    this._setEl('tgvErosions', settings.tgv.erosions);

    // Phase offset
    const phase_offset_method = settings.phase_offset_method || 'mcpc3ds';
    this._setChecked('phase_offset_enabled', phase_offset_method !== 'none');
    this._setEl('phase_offset_method', phase_offset_method === 'none' ? 'mcpc3ds' : phase_offset_method);

    // MCPC-3D-S settings
    this._setEl('mcpc3dsSigmaX', settings.mcpc3ds?.sigma?.[0] ?? 10);
    this._setEl('mcpc3dsSigmaY', settings.mcpc3ds?.sigma?.[1] ?? 10);
    this._setEl('mcpc3dsSigmaZ', settings.mcpc3ds?.sigma?.[2] ?? 5);

    // Phase unwrap method
    const unwrapping_algorithm = settings.unwrapping_algorithm || 'romeo';
    this._setEl('unwrapping_algorithm', unwrapping_algorithm);
    this._showEl('romeo_settings', unwrapping_algorithm === 'romeo');
    this._showEl('laplacian_settings', unwrapping_algorithm === 'laplacian');

    // ROMEO weight component checkboxes
    const romeoSettings = settings.romeo || {};
    this._setChecked('romeo_phase_gradient_coherence', romeoSettings.phase_gradient_coherence !== false);
    this._setChecked('romeo_mag_coherence', romeoSettings.mag_coherence !== false);
    this._setChecked('romeo_mag_weight', romeoSettings.mag_weight !== false);
    this._setEl('romeo_multi_echo_mode', romeoSettings.individual === false ? 'template' : 'individual');
    this._setChecked('romeo_correct_global', romeoSettings.correct_global !== false);
    this._setEl('romeo_template_echo', String((romeoSettings.template ?? 0) + 1));

    // Field calculation method
    const fieldCalcMethod = settings.b0_estimation || 'weighted_avg';
    this._setEl('b0_estimation', fieldCalcMethod);
    this._showEl('weighted_avg_settings', fieldCalcMethod === 'weighted_avg');
    this._showEl('linear_fit_settings', fieldCalcMethod === 'linear_fit');

    // B0 weight type
    this._setEl('b0_weight_type', settings.b0_weight_type ?? 'phase_snr');

    // Linear fit settings
    this._setChecked('linear_fit_estimate_offset', settings.linearFit?.estimate_offset ?? true);

    // Background removal method
    const bgMethod = settings.bf_algorithm;
    this._setEl('bf_algorithm', bgMethod);
    this._showEl('vsharp_settings', bgMethod === 'vsharp');
    this._showEl('sharp_settings', bgMethod === 'sharp');
    this._showEl('resharp_settings', bgMethod === 'resharp');
    this._showEl('ismv_settings', bgMethod === 'ismv');
    this._showEl('pdf_settings', bgMethod === 'pdf');
    this._showEl('lbv_settings', bgMethod === 'lbv');
    this._showEl('harperella_settings', bgMethod === 'harperella');
    this._showEl('iharperella_settings', bgMethod === 'iharperella');

    // V-SHARP settings
    this._setEl('vsharpMaxRadius', settings.vsharp.max_radius ?? defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', settings.vsharp.min_radius ?? defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', settings.vsharp.threshold);

    // RESHARP settings
    if (settings.resharp) {
      this._setEl('resharpRadius', settings.resharp.radius);
      this._setEl('resharpTikReg', settings.resharp.tik_reg);
      this._setEl('resharpTol', settings.resharp.tol);
      this._setEl('resharpMaxIter', settings.resharp.max_iter);
    }

    // HARPERELLA settings
    if (settings.harperella) {
      this._setEl('harperellaRadius', settings.harperella.radius);
      this._setEl('harperellaMaxIter', settings.harperella.max_iter);
    }

    // iHARPERELLA settings
    if (settings.iharperella) {
      this._setEl('iharperellaRadius', settings.iharperella.radius);
      this._setEl('iharperellaMaxIter', settings.iharperella.max_iter);
    }

    // iSMV settings
    this._setEl('ismv_radius', settings.ismv.radius ?? defaults.ismv_radius);
    this._setEl('ismvTol', settings.ismv.tol);
    this._setEl('ismvMaxit', settings.ismv.maxit);

    // PDF settings
    this._setEl('pdfTol', settings.pdf.tol);
    this._setEl('pdfMaxit', settings.pdf.maxit ?? defaults.pdfMaxit);

    // LBV settings
    this._setEl('lbvTol', settings.lbv.tol);
    this._setEl('lbvMaxit', settings.lbv.maxit ?? defaults.lbvMaxit);

    // Dipole inversion method
    const dipoleMethod = settings.dipole_inversion;
    this._setEl('dipole_method', dipoleMethod);
    this._showEl('tkd_settings', dipoleMethod === 'tkd');
    this._showEl('tsvd_settings', dipoleMethod === 'tsvd');
    this._showEl('tikhonov_settings', dipoleMethod === 'tikhonov');
    this._showEl('tv_settings', dipoleMethod === 'tv');
    this._showEl('rts_settings', dipoleMethod === 'rts');
    this._showEl('nltv_settings', dipoleMethod === 'nltv');
    this._showEl('medi_settings', dipoleMethod === 'medi');
    this._showEl('ndi_settings', dipoleMethod === 'ndi');
    this._showEl('fansi_settings', dipoleMethod === 'fansi');
    this._showEl('fansitgv_settings', dipoleMethod === 'fansitgv');
    this._showEl('l1qsm_settings', dipoleMethod === 'l1qsm');
    this._showEl('whqsm_settings', dipoleMethod === 'whqsm');
    this._showEl('hdqsm_settings', dipoleMethod === 'hdqsm');
    this._showEl('ilsqr_settings', dipoleMethod === 'ilsqr');

    // TKD settings
    this._setEl('tkdThreshold', settings.tkd.threshold);

    // TSVD settings
    this._setEl('tsvdThreshold', settings.tsvd.threshold);

    // Tikhonov settings
    this._setEl('tikhLambda', settings.tikhonov.lambda);
    this._setEl('tikhReg', settings.tikhonov.reg);

    // TV-ADMM settings
    this._setEl('tvLambda', settings.tv.lambda);
    this._setEl('tvMaxIter', settings.tv.max_iter);
    this._setEl('tvTol', settings.tv.tol);

    // RTS settings
    this._setEl('rtsDelta', settings.rts.delta);
    this._setEl('rtsMu', settings.rts.mu);
    this._setEl('rtsRho', settings.rts.rho);
    this._setEl('rtsMaxIter', settings.rts.max_iter);

    // NLTV settings
    this._setEl('nltvLambda', settings.nltv.lambda);
    this._setEl('nltvMu', settings.nltv.mu);
    this._setEl('nltvMaxIter', settings.nltv.max_iter);
    this._setEl('nltvTol', settings.nltv.tol);
    this._setEl('nltvNewtonMaxIter', settings.nltv.newton_max_iter);

    // NDI settings
    const ndi = settings.ndi || NDI_DEFAULTS;
    this._setEl('ndiTau', ndi.tau);
    this._setEl('ndiAlpha', ndi.alpha);
    this._setEl('ndiMaxIter', ndi.max_iter);

    // FANSI (Nonlinear TV) settings
    const fansi = settings.fansi || FANSI_DEFAULTS;
    this._setEl('fansiAlpha1', fansi.alpha1);
    this._setEl('fansiMu1', fansi.mu1);
    this._setEl('fansiMu2', fansi.mu2);
    this._setEl('fansiAlpha0', fansi.alpha0);
    this._setEl('fansiMu0', fansi.mu0);
    this._setEl('fansiMaxIter', fansi.max_iter);
    this._setEl('fansiTolUpdate', fansi.tol_update);

    // FANSI (Nonlinear TGV) settings
    const fansitgv = settings.fansitgv || FANSI_DEFAULTS;
    this._setEl('fansitgvAlpha1', fansitgv.alpha1);
    this._setEl('fansitgvMu1', fansitgv.mu1);
    this._setEl('fansitgvMu2', fansitgv.mu2);
    this._setEl('fansitgvAlpha0', fansitgv.alpha0);
    this._setEl('fansitgvMu0', fansitgv.mu0);
    this._setEl('fansitgvMaxIter', fansitgv.max_iter);
    this._setEl('fansitgvTolUpdate', fansitgv.tol_update);

    // L1-QSM settings
    const l1qsm = settings.l1qsm || L1QSM_DEFAULTS;
    this._setEl('l1qsmAlpha1', l1qsm.alpha1);
    this._setEl('l1qsmMu1', l1qsm.mu1);
    this._setEl('l1qsmMu2', l1qsm.mu2);
    this._setEl('l1qsmMu3', l1qsm.mu3);
    this._setEl('l1qsmLambda', l1qsm.lambda);
    this._setEl('l1qsmMaxIter', l1qsm.max_iter);
    this._setEl('l1qsmTolUpdate', l1qsm.tol_update);

    // WH-QSM settings
    const whqsm = settings.whqsm || WHQSM_DEFAULTS;
    this._setEl('whqsmAlpha1', whqsm.alpha1);
    this._setEl('whqsmMu1', whqsm.mu1);
    this._setEl('whqsmMu2', whqsm.mu2);
    this._setEl('whqsmBeta', whqsm.beta);
    this._setEl('whqsmMuh', whqsm.muh);
    this._setEl('whqsmMaxIter', whqsm.max_iter);
    this._setEl('whqsmTolUpdate', whqsm.tol_update);

    // HD-QSM settings
    const hdqsm = settings.hdqsm || HDQSM_DEFAULTS;
    this._setEl('hdqsmAlphaL2', hdqsm.alpha_l2);
    this._setEl('hdqsmMu1L2', hdqsm.mu1_l2);
    this._setEl('hdqsmMu2', hdqsm.mu2);
    this._setEl('hdqsmMaxIterL1', hdqsm.max_iter_l1);
    this._setEl('hdqsmMaxIterL2', hdqsm.max_iter_l2);
    this._setEl('hdqsmTolUpdate', hdqsm.tol_update);

    // MEDI settings
    this._setEl('mediLambda', settings.medi.lambda);
    this._setEl('mediPercentage', settings.medi.percentage);
    this._setEl('mediMaxIter', settings.medi.max_iter);
    this._setEl('mediCgMaxIter', settings.medi.cg_max_iter);
    this._setChecked('mediSmv', settings.medi.smv);
    this._setEl('mediSmvRadius', settings.medi.smv_radius);
    this._showEl('mediSmvRadiusGroup', settings.medi.smv);
    this._setChecked('mediMerit', settings.medi.merit);

    // iLSQR settings
    this._setEl('ilsqr_tol', settings.ilsqr?.tol || 0.01);
    this._setEl('ilsqr_max_iter', settings.ilsqr?.max_iter || 50);
  }

  _setupEventListeners() {
    // Combined method dropdown - show/hide TGV/QSMART settings
    this._on('combined_method', 'change', () => this._onCombinedMethodChange());

    // Phase offset enabled checkbox
    this._on('phase_offset_enabled', 'change', () => this._onCombinedMethodChange());

    // Bipolar correction checkbox
    this._on('bipolar_correctionEnabled', 'change', () => this._onCombinedMethodChange());

    // Unwrap method dropdown
    this._on('unwrapping_algorithm', 'change', () => this._onCombinedMethodChange());

    // Background removal method dropdown
    this._on('bf_algorithm', 'change', (e) => {
      const method = e.target.value;
      this._showEl('vsharp_settings', method === 'vsharp');
      this._showEl('sharp_settings', method === 'sharp');
      this._showEl('resharp_settings', method === 'resharp');
      this._showEl('ismv_settings', method === 'ismv');
      this._showEl('pdf_settings', method === 'pdf');
      this._showEl('lbv_settings', method === 'lbv');
      this._showEl('harperella_settings', method === 'harperella');
      this._showEl('iharperella_settings', method === 'iharperella');
    });

    // MEDI SMV checkbox toggle
    this._on('mediSmv', 'change', (e) => {
      this._showEl('mediSmvRadiusGroup', e.target.checked);
      this._onCombinedMethodChange(); // Re-check visibility
    });

    // Dipole method change
    this._on('dipole_method', 'change', (e) => {
      const method = e.target.value;
      this._showEl('tkd_settings', method === 'tkd');
      this._showEl('tsvd_settings', method === 'tsvd');
      this._showEl('tikhonov_settings', method === 'tikhonov');
      this._showEl('tv_settings', method === 'tv');
      this._showEl('rts_settings', method === 'rts');
      this._showEl('nltv_settings', method === 'nltv');
      this._showEl('medi_settings', method === 'medi');
      this._showEl('ndi_settings', method === 'ndi');
      this._showEl('fansi_settings', method === 'fansi');
      this._showEl('fansitgv_settings', method === 'fansitgv');
      this._showEl('l1qsm_settings', method === 'l1qsm');
      this._showEl('whqsm_settings', method === 'whqsm');
      this._showEl('hdqsm_settings', method === 'hdqsm');
      this._showEl('ilsqr_settings', method === 'ilsqr');
      this._onCombinedMethodChange(); // Re-check visibility for MEDI SMV
    });

    // QSMART inner inversion method change - show that algorithm's params
    this._on('qsmartInversionMethod', 'change', () => this._updateQsmartInnerVisibility());

    // QSMART MEDI SMV checkbox toggle
    this._on('qsmartMediSmv', 'change', (e) => {
      this._showEl('qsmartMediSmvRadiusGroup', e.target.checked);
    });

    // Phase offset method dropdown
    this._on('phase_offset_method', 'change', () => this._onCombinedMethodChange());

    // Field calculation method dropdown
    this._on('b0_estimation', 'change', () => this._onCombinedMethodChange());

    // ROMEO magnitude weight checkboxes - re-evaluate warnings on change
    ['romeo_mag_coherence', 'romeo_mag_weight'].forEach(id => {
      this._on(id, 'change', () => this._onCombinedMethodChange());
    });

    // BET fractional intensity slider value display
    this._on('betFractionalIntensity', 'input', (e) => {
      const valueEl = document.getElementById('betFractionalIntensityValue');
      if (valueEl) valueEl.textContent = e.target.value;
    });
  }

  _onCombinedMethodChange() {
    this.updateVisibility(this.nEchoes || 0);
  }

  // Show the parameter group matching the QSMART inner inversion selection.
  _updateQsmartInnerVisibility() {
    const inner = this._getEl('qsmartInversionMethod') || 'ilsqr';
    ['ilsqr', 'tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi'].forEach(a => {
      this._showEl(`qsmart_${a}_settings`, a === inner);
    });
    if (inner === 'medi') {
      this._showEl('qsmartMediSmvRadiusGroup', this._getChecked('qsmartMediSmv'));
    }
  }

  // DOM helper methods
  _getEl(id) {
    const el = document.getElementById(id);
    return el ? el.value : null;
  }

  _setEl(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
  }

  _getChecked(id) {
    const el = document.getElementById(id);
    return el ? el.checked : null;
  }

  _setChecked(id, checked) {
    const el = document.getElementById(id);
    if (el) el.checked = checked;
  }

  _showEl(id, show) {
    const el = document.getElementById(id);
    if (el) el.style.display = show ? 'block' : 'none';
  }

  _disableEl(id, disabled) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.opacity = disabled ? '0.4' : '';
    el.style.pointerEvents = disabled ? 'none' : '';
    for (const input of el.querySelectorAll('input, select, button')) {
      input.disabled = disabled;
    }
  }


  /**
   * Show/hide an inline warning message near a control
   * @param {string} anchorId - ID of the element to attach warning after
   * @param {string} warningId - Unique ID for the warning element
   * @param {boolean} show - Whether to show the warning
   * @param {string} message - Warning text
   * @param {string} [type='warning'] - 'warning' or 'info'
   */
  _showWarning(anchorId, warningId, show, message, type = 'warning') {
    let warning = document.getElementById(warningId);
    const anchor = document.getElementById(anchorId);

    if (show) {
      if (!warning && anchor) {
        warning = document.createElement('span');
        warning.id = warningId;
        warning.className = `validation-message ${type} inline-warning`;
        warning.innerHTML = '<span></span>';
        // If anchor is inside a <label>, insert after the label instead
        const insertAfter = anchor.parentNode.tagName === 'LABEL' ? anchor.parentNode : anchor;
        insertAfter.parentNode.insertBefore(warning, insertAfter.nextSibling);
      }
      if (warning) {
        warning.querySelector('span').textContent = message;
        warning.style.display = '';
      }
    } else if (warning) {
      warning.style.display = 'none';
    }
  }

  /**
   * Show/hide a warning banner at the top of a section (replaces _disableSection)
   * All inputs remain interactive.
   * @param {string} id - Section element ID
   * @param {boolean} hasWarning - Whether to show the warning
   * @param {string} [warningText] - Warning text
   */
  _showSectionWarning(id, hasWarning, warningText) {
    const section = document.getElementById(id);
    if (!section) return;

    const warningId = id + 'Warning';
    let warning = document.getElementById(warningId);

    if (hasWarning && warningText) {
      if (!warning) {
        warning = document.createElement('div');
        warning.id = warningId;
        warning.className = 'validation-message error inline-warning';
        warning.innerHTML = '<span></span>';
        const heading = section.querySelector('h4');
        if (heading) {
          heading.after(warning);
        } else {
          section.prepend(warning);
        }
      }
      warning.querySelector('span').textContent = warningText;
      warning.style.display = 'flex';
    } else if (warning) {
      warning.style.display = 'none';
    }
  }

  _on(id, event, handler) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, handler);
  }
}
