// Import extracted utility modules
import { createThresholdMask } from './modules/mask/ThresholdUtils.js';
import {
  parseNiftiHeader,
  isGzipped,
  isValidNifti1,
  readNiftiImageData,
  createMaskNifti,
  createNiftiHeaderFromVolume,
  createFloat64Nifti
} from './modules/file-io/NiftiUtils.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { LandingPage } from './modules/ui/LandingPage.js';
import { Tutorial, WelcomePrompt } from './modules/ui/Tutorial.js';
import { EchoNavigator } from './modules/viewer/EchoNavigator.js';
import { FileIOController, PipelineExecutor, PipelineSettingsController, MaskController, ViewerController } from './controllers/index.js';
import { DicomController } from './controllers/DicomController.js';
import { DicompareController } from 'https://dicompare.neurodesk.org/embed/DicompareController.js';
import { DicompareReportRenderer } from 'https://dicompare.neurodesk.org/embed/DicompareReportRenderer.js';
import * as QSMConfig from './app/config.js';
import { buildConfigJson, maskSectionString } from './modules/ConfigBridge.js';

// Make config available globally for backward compatibility
window.QSMConfig = QSMConfig;

/** Simple markdown → HTML for methods text (headings, paragraphs, lists). */
function renderMarkdown(md) {
  return md
    // Protect literal scientific asterisks (T2*, R2*) so they aren't paired into
    // markdown italics by the emphasis pass below. &#42; renders as a literal '*'.
    // (The raw/copyable text keeps the plain 'T2*'.)
    .replace(/(T2|R2)\*/g, '$1&#42;')
    .replace(/^## (.+)$/gm, '<h3 style="margin-top: 1em; margin-bottom: 0.3em; font-size: 1rem;">$1</h3>')
    .replace(/^# (.+)$/gm, '<h2 style="margin-top: 0; margin-bottom: 0.5em; font-size: 1.1rem;">$1</h2>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul style="margin: 0.3em 0; padding-left: 1.5em;">${m}</ul>`)
    .replace(/\n\n/g, '</p><p style="margin: 0.5em 0;">')
    .replace(/^\s*</, '<')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

class QSMApp {
  constructor() {
    // Config is required - no fallbacks
    const cfg = window.QSMConfig;

    this.nv = new window.Niivue({
      ...cfg.VIEWER_CONFIG,
      onLocationChange: (data) => {
        document.getElementById("intensity").innerHTML = data.string;
      }
    });
    this.currentFile = null;
    this.threshold = 75;
    this.progress = 0;

    // Smooth progress animation state
    this.targetProgress = 0;
    this.animatedProgress = 0;
    this.progressAnimationId = null;
    this.lastAnimationTime = 0;
    this.progressAnimationSpeed = cfg.PROGRESS_CONFIG.animationSpeed;

    // Controllers (initialized in init() after DOM ready)
    this.fileIOController = null;
    this.pipelineExecutor = null;

    // Mask threshold (percentage of max magnitude)
    this.maskThreshold = cfg.MASK_CONFIG.defaultThreshold;
    this.magnitudeData = null;
    this.magnitudeMax = 0;

    // Mask editing state
    this.currentMaskData = null;
    this.originalMaskData = null;
    this.maskDims = null;
    this.voxelSize = null;

    // Drawing state
    this.drawingEnabled = false;
    this.brushMode = 'add';
    this.brushSize = cfg.MASK_CONFIG.defaultBrushSize;
    this.savedCrosshairWidth = cfg.VIEWER_CONFIG.crosshairWidth;

    // Pipeline settings from config
    this.pipelineSettings = JSON.parse(JSON.stringify(cfg.PIPELINE_DEFAULTS));

    // Mask operations history for command generation
    this.maskOpsHistory = [];

    // BET settings from config
    this.betSettings = { ...cfg.BET_DEFAULTS };

    // Mask preparation settings from config
    this.maskPrepSettings = { ...cfg.MASK_PREP_DEFAULTS, prepared: false };
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;

    // Echo navigation state
    this.currentEchoIndex = 0;
    this.currentViewType = null;

    // Controllers (initialized in init() after DOM ready)
    this.pipelineSettingsController = null;
    this.maskController = null;
    this.viewerController = null;

    // Modal managers (initialized in init() after DOM ready)
    this.betModal = null;
    this.aboutModal = null;
    this.citationsModal = null;
    this.privacyModal = null;

    this.init();
  }

  // Getters for backward compatibility - delegates to PipelineExecutor
  get pipelineRunning() {
    return this.pipelineExecutor?.isRunning() || false;
  }

  get worker() {
    return this.pipelineExecutor?.getWorker() || null;
  }

  get workerReady() {
    return this.pipelineExecutor?.isReady() || false;
  }

  get results() {
    return this.pipelineExecutor?.getResults() || {};
  }

  get stageOrder() {
    return this.pipelineExecutor?.getStageOrder() || [];
  }

  async init() {
    // Display version in header
    const versionEl = document.getElementById('appVersion');
    if (versionEl && window.QSMConfig?.VERSION) {
      versionEl.textContent = `v${window.QSMConfig.VERSION}`;
    }
    const landingVersionEl = document.getElementById('landingVersion');
    if (landingVersionEl && window.QSMConfig?.VERSION) {
      landingVersionEl.textContent = `v${window.QSMConfig.VERSION}`;
    }

    // Set up landing/tutorial first so the welcome overlay is always
    // dismissable, even if a later init step (e.g. the WebGL viewer) fails.
    this._setupOnboarding();

    // Initialize FileIOController first (other controllers depend on it)
    this.fileIOController = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFilesChanged: () => this._onBucketsChanged(),
      onMagnitudeFilesChanged: (files) => this._onMagnitudeFilesChanged(files),
      onPhaseFilesChanged: (files) => this._onPhaseFilesChanged(files)
    });
    this.fileIOController.setupEchoTagify();

    await this.setupViewer();
    this.setupUIControls();
    this.setupEventListeners();
    this.syncSidebarFromSettings();
    this.updateDownloadButtons();

    // Initialize mask file list via controller
    this.fileIOController.updateFileList('mask', []);

    // Initialize masking controls state (disabled until Prepare is clicked)
    this.updateMaskingControlsState();

    // Sync mask prep settings with actual UI state
    const sourceSelect = document.getElementById('maskInputSource');
    const biasCheckbox = document.getElementById('applyBiasCorrection');
    if (sourceSelect) this.maskPrepSettings.source = sourceSelect.value;
    if (biasCheckbox) this.maskPrepSettings.biasCorrection = biasCheckbox.checked;

    // Initialize controllers
    const pipelineModal = document.getElementById('pipelineSettingsModal');
    if (pipelineModal) {
      this.pipelineSettingsController = new PipelineSettingsController(pipelineModal);
    }

    // Initialize pipeline executor (before mask controller, provides worker)
    this.pipelineExecutor = new PipelineExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      onStageData: (data) => this._onStageData(data),
      onPipelineComplete: () => this._onPipelineComplete(),
      onPipelineError: () => this._onPipelineError(),
      config: window.QSMConfig
    });

    // Initialize mask controller
    this.maskController = new MaskController({
      nv: this.nv,
      getWorker: () => this.pipelineExecutor?.getWorker(),
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      initializeWorker: () => this.pipelineExecutor?.initialize(),
      config: window.QSMConfig
    });

    // Initialize viewer controller
    this.viewerController = new ViewerController({
      nv: this.nv,
      getMultiEchoFiles: () => this.fileIOController?.getMultiEchoFiles() || { magnitude: [], phase: [], json: [], combinedMagnitude: null, combinedPhase: null },
      updateOutput: (msg) => this.updateOutput(msg),
      showOverlayControl: (show) => this.showOverlayControl(show),
      updateDownloadVolumeButton: () => this.updateDownloadVolumeButton()
    });

    // Initialize dicompare controller
    this.dicompareController = new DicompareController({
      schemaUrl: 'https://dicompare.neurodesk.org/schemas/QSM_Consensus_Guidelines_v1.0.json',
      updateOutput: (msg) => this.updateOutput(msg)
    });
    this.dicompareRenderer = new DicompareReportRenderer();

    // Initialize DICOM controller
    this.dicomController = new DicomController({
      updateOutput: (msg) => this.updateOutput(msg),
      onConversionComplete: (classified) => this._onDicomConversionComplete(classified),
      onFilesRetained: (files) => this._onDicomFilesRetained(files)
    });

    // Initialize modal managers
    this.betModal = new ModalManager('betSettingsModal');
    this.commandPreviewModal = new ModalManager('commandPreviewModal');
    this.aboutModal = new ModalManager('aboutModal');
    this.citationsModal = new ModalManager('citationsModal');
    this.privacyModal = new ModalManager('privacyModal');
    this.dicompareModal = new ModalManager('dicompareModal');

    // Start loading WASM in the background immediately
    this.pipelineExecutor.initialize();
  }

  /**
   * Wire up the landing overlay, welcome-tour prompt and guided tutorial.
   * On first visit the landing page is shown; launching from it may kick off
   * the tour. The tour is always re-launchable from the header "Guide" button.
   */
  _setupOnboarding() {
    this.tutorial = new Tutorial(this._buildTourSteps());

    this.welcomePrompt = new WelcomePrompt({
      onStart: () => this.tutorial.start(),
    });

    this.landingPage = new LandingPage({
      // Normal launch: offer the tour once (unless the user opted out).
      onLaunch: () => {
        if (!this.welcomePrompt.isDismissed()) {
          this.welcomePrompt.open();
        }
      },
      // "Take a guided tour" button: start immediately.
      onLaunchTour: () => this.tutorial.start(),
    });

    // Header "Guide" button re-runs the tour on demand.
    document.getElementById('openGuide')?.addEventListener('click', () => {
      if (!this.tutorial.isRunning()) this.tutorial.start();
    });

    // Clicking the logo returns to the welcome page.
    document.getElementById('appLogo')?.addEventListener('click', () => {
      this.tutorial.stop();
      this.landingPage.show();
    });
  }

  /** Expand a collapsed sidebar accordion section by id, and scroll to it. */
  _expandSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    section.classList.remove('hidden');
    section.classList.remove('collapsed');
  }

  /** Step definitions for the guided tour (targets existing sidebar elements). */
  _buildTourSteps() {
    return [
      {
        selector: '#inputSection',
        title: 'Start with your data',
        body: 'Drop NIfTI or DICOM magnitude and phase files here — or click "Load example data" to try QSMbly instantly.',
        placement: 'right',
        onEnter: () => this._expandSection('inputSection'),
        // Advance once files have been loaded (the triage panel appears).
        waitFor: () => {
          const t = document.getElementById('fileTriage');
          return !!t && t.style.display !== 'none';
        },
      },
      {
        selector: '#paramsSection',
        title: 'Set acquisition parameters',
        body: 'Enter the echo times and field strength for your scan. These auto-fill from JSON sidecars when available.',
        placement: 'right',
        onEnter: () => this._expandSection('paramsSection'),
      },
      {
        selector: '#maskSection',
        title: 'Create a brain mask',
        body: 'Pick a masking source and click "Prepare Input" to generate a brain mask — or upload your own.',
        placement: 'right',
        onEnter: () => this._expandSection('maskSection'),
      },
      {
        selector: '#pipelineSection',
        title: 'Choose your pipeline',
        body: 'Select an algorithm for each stage: phase unwrapping, background removal and dipole inversion. Use "Advanced" for fine-grained control.',
        placement: 'right',
        onEnter: () => this._expandSection('pipelineSection'),
      },
      {
        selector: '#runPipelineSidebar',
        title: 'Run the reconstruction',
        body: 'Click "Start QSM" to run the full pipeline. Progress appears at the bottom of the sidebar.',
        placement: 'right',
      },
      {
        selector: '.app-main',
        title: 'Explore your results',
        body: 'Your susceptibility map and every intermediate stage appear here. Pan, zoom and adjust contrast interactively.',
        placement: 'left',
      },
      {
        selector: '#exportBar',
        title: 'Reproduce & cite your run',
        body: 'When you\'re happy with a pipeline, grab the equivalent QSMxT command line to batch-process on your own machine, or a ready-to-paste methods paragraph with citations. That\'s it — enjoy!',
        placement: 'top',
      },
    ];
  }

  // Pipeline executor callbacks
  _onStageData(data) {
    // displayNow defaults to true for backward compatibility
    const displayNow = data.displayNow !== false;
    if (displayNow) {
      this.displayLiveStageData(data);
    } else {
      this.cacheStageData(data);
    }
  }

  _onPipelineComplete() {
    this.showStageButtons();
    document.getElementById('cancelPipeline').disabled = true;
    document.getElementById('runSWI').disabled = false;
    document.getElementById('runT2starR2star').disabled = false;
    this.updateEchoInfo();
  }

  _onPipelineError() {
    document.getElementById('cancelPipeline').disabled = true;
    document.getElementById('runSWI').disabled = false;
    document.getElementById('runT2starR2star').disabled = false;
    this.updateEchoInfo();
  }

  async setupViewer() {
    await this.nv.attachTo("gl1");
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);

    // Guard against scroll/click events before any volume is loaded.
    // NiiVue registers listeners on attachTo but crashes if volumes[] is empty.
    const origMoveCrosshair = this.nv.moveCrosshairInVox.bind(this.nv);
    this.nv.moveCrosshairInVox = (...args) => {
      if (!this.nv.volumes || this.nv.volumes.length === 0) return;
      return origMoveCrosshair(...args);
    };

    this.nv.drawScene();
  }

  setupUIControls() {
    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
      sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
      });
    }

    // Console toggle
    const consoleHeader = document.querySelector('.console-header');
    const consoleEl = document.getElementById('console');
    if (consoleHeader && consoleEl) {
      consoleHeader.addEventListener('click', () => {
        consoleEl.classList.toggle('collapsed');
      });
    }

    // Stage tab switching
    const stageTabs = document.querySelectorAll('.stage-tab');
    stageTabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-download')) return;
        stageTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
      });
    });
  }

  setProgress(value, text = null) {
    this.progress = value;
    this.targetProgress = value;

    const textEl = document.getElementById('progressText');
    if (textEl) textEl.textContent = text || `${Math.round(value * 100)}%`;

    // Update progress bar immediately for accurate feedback
    this.animatedProgress = value;
    this.updateProgressBar();

    // Stop any running animation since we update immediately
    this.stopProgressAnimation();
  }

  animateProgress() {
    const now = performance.now();
    const deltaTime = (now - this.lastAnimationTime) / 1000; // Convert to seconds
    this.lastAnimationTime = now;

    // Move animated progress toward target, but don't exceed it
    if (this.animatedProgress < this.targetProgress) {
      // Calculate how much to move based on time elapsed
      const increment = this.progressAnimationSpeed * deltaTime;
      this.animatedProgress = Math.min(this.animatedProgress + increment, this.targetProgress);
      this.updateProgressBar();
    }
    // If animated progress has caught up to target, we just wait (pause)
    // The bar will resume moving when a new setProgress call increases targetProgress

    // Continue animation loop if not complete
    if (this.targetProgress < 1 && this.targetProgress > 0) {
      this.progressAnimationId = requestAnimationFrame(() => this.animateProgress());
    } else {
      this.progressAnimationId = null;
    }
  }

  updateProgressBar() {
    const pct = `${this.animatedProgress * 100}%`;
    const fill = document.getElementById('progressFill');
    if (fill) fill.style.width = pct;
    const mobileFill = document.getElementById('mobileProgressFill');
    if (mobileFill) mobileFill.style.width = pct;
  }

  stopProgressAnimation() {
    if (this.progressAnimationId) {
      cancelAnimationFrame(this.progressAnimationId);
      this.progressAnimationId = null;
    }
  }

  setupEventListeners() {
    // Mobile tab bar
    this._setupMobileTabs();

    // Info tooltips
    this._setupInfoTooltips();

    // Unified file input
    this._setupUnifiedDropZone();

    // Load example data button
    document.getElementById('loadExampleData')?.addEventListener('click', () => this._loadExampleData());

    // Field map units dropdown
    document.getElementById('fieldMapUnits')?.addEventListener('change', () => {
      this.updateInputParamsVisibility();
      this.updateEchoInfo();
    });

    // Centralized mask file input (in Masking section)
    document.getElementById('maskFiles')?.addEventListener('change', (e) => {
      this.fileIOController.handleMaskInput(e);
      this.updateMaskSectionState();
      this.updateEchoInfo();
    });

    // dicompare report
    document.getElementById('dicompareReportBtn')?.addEventListener('click', () => this.runDicompareReport());
    document.getElementById('closeDicompare')?.addEventListener('click', () => this.dicompareModal?.close());
    document.getElementById('closeDicompare2')?.addEventListener('click', () => this.dicompareModal?.close());
    document.getElementById('dicomparePrint')?.addEventListener('click', () => this.printDicompareReport());

    // Preview buttons (mask only - magnitude/phase/fieldmap previews now in triage)
    document.getElementById('vis_mask')?.addEventListener('click', () => this.visualizeMaskFile());

    // Sidebar pipeline dropdowns
    this.setupSidebarDropdownListeners();

    // Processing buttons
    document.getElementById('openPipelineSettings').addEventListener('click', () => this.openPipelineSettingsModal());
    document.getElementById('cancelPipeline')?.addEventListener('click', () => this.cancelPipeline());

    // Echo navigation
    document.getElementById('echoPrev')?.addEventListener('click', () => this.navigateEcho(-1));
    document.getElementById('echoNext')?.addEventListener('click', () => this.navigateEcho(1));

    // Note: Stage show/download buttons are now created dynamically in addStageButton()

    // Mask threshold slider with debounce
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        this.maskThreshold = parseInt(e.target.value);
        document.getElementById('thresholdLabel').textContent = `Threshold (${this.maskThreshold}%)`;

        // Debounce the mask preview update
        if (this.maskUpdateTimeout) {
          clearTimeout(this.maskUpdateTimeout);
        }
        this.maskUpdateTimeout = setTimeout(() => {
          if (this.magnitudeData && !this.maskUpdating) {
            this.updateMaskPreview();
          }
        }, 150);
      });
    }

    // Preview mask button - shows Robust/Manual sub-buttons
    const previewMaskBtn = document.getElementById('previewMask');
    if (previewMaskBtn) {
      previewMaskBtn.addEventListener('click', () => {
        document.getElementById('thresholdModeButtons').style.display = '';
      });
    }

    // Threshold Robust button - Otsu + auto-refinement
    document.getElementById('thresholdRobust')?.addEventListener('click', async () => {
      document.getElementById('thresholdModeButtons').style.display = 'none';
      await this.previewMask();
      this.maskOpsHistory = ['threshold:otsu'];
      this.updateOutput("Applying robust refinement (dilate, fill holes, erode x2)...");
      this.dilateMask3D();
      this._pushMaskOp('dilate');
      this.fillHoles3D();
      this.maskOpsHistory.push('fill-holes:0');
      this.erodeMask3D();
      this._pushMaskOp('erode');
      this.erodeMask3D();
      this._pushMaskOp('erode');
      await this.displayCurrentMask();
      this.updateOutput("Robust mask complete");
    });

    // Threshold Manual button - Otsu + slider
    document.getElementById('thresholdManual')?.addEventListener('click', async () => {
      document.getElementById('thresholdModeButtons').style.display = 'none';
      await this.previewMask();
      this.maskOpsHistory = ['threshold:otsu'];
      const sliderGroup = document.getElementById('thresholdSliderGroup');
      if (sliderGroup) sliderGroup.style.display = '';
      this.setThresholdSliderEnabled(true);
    });

    // BET brain extraction button - opens settings modal
    document.getElementById('runBET')?.addEventListener('click', () => this.openBetSettingsModal());

    // Auto threshold button (Otsu)
    document.getElementById('autoThreshold')?.addEventListener('click', () => this.autoDetectThreshold());

    // Mask Input Preparation
    document.getElementById('maskInputSource')?.addEventListener('change', (e) => {
      const isCustom = e.target.value === 'custom';
      this.maskPrepSettings.source = e.target.value;
      this.maskPrepSettings.prepared = false;
      this.updateMagnitudePrepSection();
      this.updatePrepareButtonState();

      // Show/hide custom mask upload vs prepare button
      const maskUploadGroup = document.getElementById('maskUploadGroup');
      const prepareMaskInput = document.getElementById('prepareMaskInput');
      if (maskUploadGroup) maskUploadGroup.style.display = isCustom ? '' : 'none';
      if (prepareMaskInput) prepareMaskInput.style.display = isCustom ? 'none' : '';

      // Hide bias correction for phase quality map and custom mask (not applicable)
      const biasCorrectionGroup = document.getElementById('biasCorrectionGroup');
      if (biasCorrectionGroup) {
        biasCorrectionGroup.style.display = (e.target.value === 'phase_quality' || isCustom) ? 'none' : '';
      }

      this.updateMaskSectionState();
    });

    document.getElementById('applyBiasCorrection')?.addEventListener('change', (e) => {
      this.maskPrepSettings.biasCorrection = e.target.checked;
      this.maskPrepSettings.prepared = false;
      console.log('Bias correction checkbox changed:', e.target.checked);
      this.updatePrepareButtonState();
    });

    document.getElementById('prepareMaskInput')?.addEventListener('click', () => {
      this.prepareMaskInput();
    });

    // Pipeline settings modal
    document.getElementById('closePipelineSettings')?.addEventListener('click', () => this.closePipelineSettingsModal());
    document.getElementById('resetPipelineSettings')?.addEventListener('click', () => this.resetPipelineSettings());
    document.getElementById('savePipelineSettings')?.addEventListener('click', () => this.savePipelineSettings());
    document.getElementById('runPipelineSidebar')?.addEventListener('click', () => this.runPipelineFromSidebar());
    document.getElementById('previewCommand')?.addEventListener('click', () => this.showCommandPreview());
    document.getElementById('closeCommandPreview')?.addEventListener('click', () => this.commandPreviewModal?.close());
    document.getElementById('closeCommandPreviewBtn')?.addEventListener('click', () => this.commandPreviewModal?.close());
    document.getElementById('copyCommand')?.addEventListener('click', () => this.copyCommandToClipboard());
    document.getElementById('downloadSettingsToml')?.addEventListener('click', () => this.downloadSettingsToml());
    document.getElementById('exportTabCommand')?.addEventListener('click', () => this.switchExportTab('command'));
    document.getElementById('exportTabMethods')?.addEventListener('click', () => this.switchExportTab('methods'));
    document.getElementById('exportCommand')?.addEventListener('click', () => { this.showCommandPreview(); this.switchExportTab('command'); });
    document.getElementById('exportMethods')?.addEventListener('click', () => { this.showCommandPreview(); this.switchExportTab('methods'); });
    document.getElementById('openSwiSettings')?.addEventListener('click', () => {
      document.getElementById('swiSettingsModal')?.classList.add('active');
    });
    document.getElementById('closeSwiSettings')?.addEventListener('click', () => {
      document.getElementById('swiSettingsModal')?.classList.remove('active');
    });
    document.getElementById('closeSwiSettings2')?.addEventListener('click', () => {
      document.getElementById('swiSettingsModal')?.classList.remove('active');
    });

    document.getElementById('runSWI')?.addEventListener('click', () => {
      // Sync sidebar SWI settings to pipeline settings before running
      const scaling = document.getElementById('sidebarSwiScaling')?.value || 'tanh';
      const strength = parseFloat(document.getElementById('sidebarSwiStrength')?.value) || 4;
      const mip_window = parseInt(document.getElementById('sidebarSwiMipWindow')?.value) || 7;
      const hp_sigmaX = parseFloat(document.getElementById('sidebarSwiHpSigmaX')?.value) || 4;
      const hp_sigmaY = parseFloat(document.getElementById('sidebarSwiHpSigmaY')?.value) || 4;
      const hp_sigmaZ = parseFloat(document.getElementById('sidebarSwiHpSigmaZ')?.value) || 0;
      if (this.pipelineSettings.swi) {
        this.pipelineSettings.swi.scaling = scaling;
        this.pipelineSettings.swi.strength = strength;
        this.pipelineSettings.swi.mip_window = mip_window;
        this.pipelineSettings.swi.hp_sigma = [hp_sigmaX, hp_sigmaY, hp_sigmaZ];
      }
      this.runSWI();
    });
    document.getElementById('runT2starR2star')?.addEventListener('click', () => this.runT2starR2star());

    // BET settings modal
    document.getElementById('closeBetSettings')?.addEventListener('click', () => this.betModal?.close());
    document.getElementById('resetBetSettings')?.addEventListener('click', () => this.resetBetSettings());
    document.getElementById('runBetWithSettings')?.addEventListener('click', () => this.runBetWithSettings());

    // About modal
    document.getElementById('openAbout')?.addEventListener('click', () => {
      const cfg = window.QSMConfig;
      const appVer = document.getElementById('aboutAppVersion');
      if (appVer && cfg?.VERSION) appVer.textContent = `v${cfg.VERSION}`;
      const coreVer = document.getElementById('aboutCoreVersion');
      if (coreVer && cfg?.QSM_RS_VERSION) coreVer.textContent = `v${cfg.QSM_RS_VERSION}`;
      this.aboutModal?.open();
    });
    document.getElementById('closeAbout')?.addEventListener('click', () => this.aboutModal?.close());

    // Citations modal
    document.getElementById('openCitations')?.addEventListener('click', () => this.citationsModal?.open());
    document.getElementById('closeCitations')?.addEventListener('click', () => this.citationsModal?.close());

    // Privacy modal
    document.getElementById('openPrivacy')?.addEventListener('click', () => this.privacyModal?.open());
    document.getElementById('closePrivacy')?.addEventListener('click', () => this.privacyModal?.close());

    // BET fractional intensity slider value display
    document.getElementById('betFractionalIntensity')?.addEventListener('input', (e) => {
      document.getElementById('betFractionalIntensityValue').textContent = e.target.value;
    });

    // Overlay opacity slider
    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const opacity = parseInt(e.target.value) / 100;
        document.getElementById('overlayOpacityValue').textContent = `${e.target.value}%`;
        this.updateOverlayOpacity(opacity);
      });
    }

    // Download current volume button
    document.getElementById('downloadCurrentVolume')?.addEventListener('click', () => {
      this.downloadCurrentVolume();
    });

    // Screenshot button
    document.getElementById('screenshotViewer')?.addEventListener('click', () => {
      this.saveScreenshot();
    });

    // Clear all results button
    document.getElementById('clearAllResults')?.addEventListener('click', () => {
      this.clearAllResults();
    });

    // Close modals on overlay click (BET and Citations handled by ModalManager)
    document.getElementById('pipelineSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'pipelineSettingsModal') this.closePipelineSettingsModal();
    });

    // Note: Pipeline settings form event listeners are now handled by PipelineSettingsController

    // Morphological operation buttons
    document.getElementById('fillHoles')?.addEventListener('click', async () => {
      this.updateOutput("Filling holes in mask...");
      this.fillHoles3D();
      this.maskOpsHistory.push('fill-holes:0');
      await this.displayCurrentMask();
      this.updateOutput("Holes filled");
    });

    document.getElementById('erodeMask')?.addEventListener('click', async () => {
      this.updateOutput("Eroding mask...");
      this.erodeMask3D();
      this._pushMaskOp('erode');
      await this.displayCurrentMask();
      this.updateOutput("Mask eroded");
    });

    document.getElementById('dilateMask')?.addEventListener('click', async () => {
      this.updateOutput("Dilating mask...");
      this.dilateMask3D();
      this._pushMaskOp('dilate');
      await this.displayCurrentMask();
      this.updateOutput("Mask dilated");
    });

    document.getElementById('resetMask')?.addEventListener('click', async () => {
      this.updateOutput("Clearing mask...");
      this.maskOpsHistory = [];
      await this.clearMask();
      this.updateOutput("Mask cleared. Choose Threshold or BET to create a new mask.");
    });

    // Drawing controls
    document.getElementById('enableDrawing')?.addEventListener('click', async () => {
      await this.toggleDrawingMode();
    });

    document.getElementById('brushAdd')?.addEventListener('click', () => {
      this.setBrushMode('add');
    });

    document.getElementById('brushRemove')?.addEventListener('click', () => {
      this.setBrushMode('remove');
    });

    document.getElementById('brushSize')?.addEventListener('input', (e) => {
      this.setBrushSize(parseInt(e.target.value));
      document.getElementById('brushSizeValue').textContent = this.brushSize;
    });

    document.getElementById('brush3D')?.addEventListener('change', (e) => {
      this.setBrush3D(e.target.checked);
    });

    document.getElementById('brushShapeSquare')?.addEventListener('click', () => {
      this.setBrushShape('square');
    });

    document.getElementById('brushShapeCircle')?.addEventListener('click', () => {
      this.setBrushShape('circle');
    });

    document.getElementById('undoDraw')?.addEventListener('click', () => {
      this.nv.drawUndo();
    });

    document.getElementById('applyDrawing')?.addEventListener('click', async () => {
      await this.applyDrawingToMask();
    });
  }

  // Passthrough for backward compatibility (HTML onclick uses app.removeFile)
  removeFile(type, index) {
    this.fileIOController.removeFile(type, index);
  }

  // ==================== Unified File Input ====================

  /**
   * Set up hover-positioned info tooltips for all .info-icon elements.
   */
  _setupMobileTabs() {
    const container = document.querySelector('.app-container');
    const tabs = document.querySelectorAll('.mobile-tab');
    if (!container || tabs.length === 0) return;

    // Set initial state
    container.setAttribute('data-mobile-tab', 'controls');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        container.setAttribute('data-mobile-tab', tabName);
        tabs.forEach(t => t.classList.toggle('active', t === tab));

        // Trigger NiiVue resize when switching to viewer
        if (tabName === 'viewer' && this.nv) {
          requestAnimationFrame(() => this.nv.resizeListener());
        }
      });
    });
  }

  _setupInfoTooltips() {
    document.querySelectorAll('.info-icon').forEach(icon => {
      const tooltip = icon.querySelector('.info-tooltip');
      if (!tooltip) return;

      icon.addEventListener('mouseenter', () => {
        tooltip.style.display = 'block';
        const iconRect = icon.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();

        let top = iconRect.top - tipRect.height - 6;
        let left = iconRect.left + iconRect.width / 2 - tipRect.width / 2;

        if (top < 4) top = iconRect.bottom + 6;
        left = Math.max(4, Math.min(left, window.innerWidth - tipRect.width - 4));

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
      });

      icon.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    });
  }

  /**
   * Set up the unified drop zone for all file types (NIfTI, DICOM, JSON).
   */
  _setupUnifiedDropZone() {
    const dropZone = document.getElementById('unifiedDrop');
    const fileInput = document.getElementById('unifiedFiles');
    if (!dropZone || !fileInput) return;

    // File input change
    fileInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      if (files.length > 0) this._handleUnifiedFiles(files);
      fileInput.value = '';
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
      if (!dropZone.contains(e.relatedTarget)) {
        dropZone.classList.remove('dragover');
      }
    });

    dropZone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');

      const items = e.dataTransfer.items;
      const files = [];
      let hasDirs = false;

      // Check for directory entries (DICOM folders)
      if (items) {
        for (const item of items) {
          const entry = item.webkitGetAsEntry?.();
          if (entry?.isDirectory) {
            hasDirs = true;
            break;
          }
        }
      }

      if (hasDirs) {
        // Directory drop → route to DICOM conversion
        this._showDicomStatus('Converting...');
        await this.dicomController.convertDropItems(items);
        this._hideDicomStatus();
        return;
      }

      // Regular files
      for (const file of e.dataTransfer.files) {
        files.push(file);
      }
      if (files.length > 0) this._handleUnifiedFiles(files);
    });
  }

  /**
   * Handle files from the unified drop zone.
   * Routes DICOM files to DicomController, NIfTI/JSON to FileIOController.
   */
  async _handleUnifiedFiles(files) {
    const dicomFiles = [];
    const niftiJsonFiles = [];

    for (const file of files) {
      const name = file.name.toLowerCase();
      if (name.endsWith('.dcm') || (!name.includes('.') && file.size > 1000)) {
        // .dcm or extensionless files (common for DICOM)
        dicomFiles.push(file);
      } else {
        niftiJsonFiles.push(file);
      }
    }

    // Route DICOM files to conversion pipeline
    if (dicomFiles.length > 0) {
      this._showDicomStatus('Converting...');
      await this.dicomController.convertFiles(dicomFiles);
      this._hideDicomStatus();
    }

    // Route NIfTI/JSON files to auto-categorized buckets
    if (niftiJsonFiles.length > 0) {
      this.fileIOController.addFiles(niftiJsonFiles);

      // Process JSON sidecars
      const jsonFiles = niftiJsonFiles.filter(f => f.name.toLowerCase().endsWith('.json'));
      if (jsonFiles.length > 0) {
        await this.fileIOController.processJsonFiles(jsonFiles);
      }

      // Auto-advance to masking section if data looks complete
      this._autoAdvanceToMasking();
    }
  }

  /**
   * If magnitude and phase are balanced, JSONs present, and field strength + echo times populated,
   * auto-open the masking section.
   */
  _autoAdvanceToMasking() {
    const b = this.fileIOController.buckets;
    const nMag = b.magnitude.length;
    const nPhase = b.phase.length;
    const nJson = b.json?.length || 0;

    // Need equal mag/phase, at least one of each, and matching JSON count
    if (nMag === 0 || nPhase === 0 || nMag !== nPhase) return;
    if (nJson < nPhase) return;

    // Check field strength is populated
    const fieldInput = document.getElementById('magField');
    if (!fieldInput || !fieldInput.value || parseFloat(fieldInput.value) <= 0) return;

    // Check echo times are populated
    const echoTagify = this.fileIOController.echoTagify;
    if (!echoTagify || echoTagify.value.length < nPhase) return;

    // All good — collapse input, open masking
    const inputSection = document.getElementById('inputSection');
    const maskSection = document.getElementById('maskSection');
    if (inputSection && maskSection) {
      inputSection.classList.add('collapsed');
      document.getElementById('paramsSection')?.classList.add('collapsed');
      maskSection.classList.remove('collapsed');
    }
  }

  /**
   * Fetch example data from GitHub Release and load it.
   */
  async _loadExampleData() {
    const btn = document.getElementById('loadExampleData');
    const { baseUrl, files } = QSMConfig.EXAMPLE_DATA;

    btn.disabled = true;
    btn.textContent = 'Downloading example data...';
    this.updateOutput('Downloading example data...');
    this.setProgress(0, 'Downloading example data...');

    try {
      // First, issue HEAD requests in parallel to learn total size
      const headResponses = await Promise.all(files.map(async (name) => {
        const resp = await fetch(`${baseUrl}/${name}`, { method: 'HEAD' });
        const len = parseInt(resp.headers.get('Content-Length') || '0', 10);
        return { name, size: len };
      }));
      const totalBytes = headResponses.reduce((sum, h) => sum + h.size, 0);
      const perFileBytes = new Array(files.length).fill(0);
      let downloadedBytes = 0;

      const updateProgress = () => {
        downloadedBytes = perFileBytes.reduce((s, b) => s + b, 0);
        const frac = totalBytes > 0 ? downloadedBytes / totalBytes : 0;
        const mb = (downloadedBytes / 1e6).toFixed(1);
        const totalMb = (totalBytes / 1e6).toFixed(1);
        this.setProgress(frac, `Downloaded ${mb} / ${totalMb} MB`);
      };

      const fetched = await Promise.all(files.map(async (name, i) => {
        const resp = await fetch(`${baseUrl}/${name}`);
        if (!resp.ok) throw new Error(`Failed to fetch ${name}: ${resp.status}`);

        const reader = resp.body.getReader();
        const chunks = [];
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          perFileBytes[i] += value.byteLength;
          updateProgress();
        }
        const blob = new Blob(chunks);
        return new File([blob], name);
      }));

      this.updateOutput(`Downloaded ${fetched.length} files. Loading...`);
      await this._handleUnifiedFiles(fetched);
      this.setProgress(0, '');
      this.updateOutput('Example data loaded successfully.');
    } catch (err) {
      this.setProgress(0, '');
      this.updateOutput(`Error loading example data: ${err.message}`);
      console.error('Example data load failed:', err);
    } finally {
      btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Load example data';
    }
  }

  /**
   * Central state handler — called whenever bucket contents change.
   * Replaces scattered switchInputMode/updateEchoInfo calls.
   */
  _onBucketsChanged() {
    // Render file triage UI
    this._renderFileTriage();

    // Update input params visibility (echo times, field strength, units)
    this.updateInputParamsVisibility();

    // Update magnitude prep and masking sections
    this.updateMagnitudePrepSection();
    this.updateMaskInputSourceOptions();
    this.updateMaskSectionState();

    // Update pipeline settings visibility
    const mode = this.fileIOController.getInputMode();
    if (this.pipelineSettingsController) {
      this.pipelineSettingsController.setInputMode(mode);
    }
    this.updateSidebarDropdownVisibility(true);
    this.updatePipelineSectionForMode(mode);

    // Update run button state
    this.updateEchoInfo();

    // Update drop zone label
    const hasFiles = Object.values(this.fileIOController.buckets).some(b => b.length > 0);
    const dropLabel = document.querySelector('#unifiedDrop .file-drop-label span');
    if (dropLabel) {
      dropLabel.textContent = hasFiles ? 'Drop more files' : 'Drop NIfTI, DICOM, or JSON files';
    }
    const dropZone = document.getElementById('unifiedDrop');
    if (dropZone) dropZone.classList.toggle('has-files', hasFiles);

    // Disable example data button when files are loaded
    const exampleBtn = document.getElementById('loadExampleData');
    if (exampleBtn) exampleBtn.disabled = hasFiles;

    // Update validation messages
    this._updateInputValidation();
  }

  /**
   * Render the interactive file triage UI with draggable file cards.
   * Generalizes the old _renderDicomTriage for all bucket types.
   */
  _renderFileTriage() {
    const container = document.getElementById('fileTriage');
    if (!container) return;

    const buckets = this.fileIOController.buckets;
    const hasAnyFiles = Object.values(buckets).some(b => b.length > 0);

    if (!hasAnyFiles) {
      container.style.display = 'none';
      container.innerHTML = '';
      return;
    }

    container.style.display = '';

    const mode = this.fileIOController.getInputMode();
    const eyeSvg = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
    const gripSvg = '<svg viewBox="0 0 12 24" width="6" height="12" fill="currentColor"><circle cx="3" cy="4" r="1.5"/><circle cx="9" cy="4" r="1.5"/><circle cx="3" cy="12" r="1.5"/><circle cx="9" cy="12" r="1.5"/><circle cx="3" cy="20" r="1.5"/><circle cx="9" cy="20" r="1.5"/></svg>';
    const deleteSvg = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

    // Bucket definitions
    const allBucketDefs = {
      magnitude: { label: 'Magnitude', canPreview: true },
      phase:     { label: 'Phase', canPreview: true },
      totalField:{ label: 'Total Field Map', canPreview: true },
      localField:{ label: 'Local Field Map', canPreview: true },
      json:      { label: 'JSON Sidecars', canPreview: false },
      extra:     { label: 'Uncategorized', canPreview: false }
    };

    // The three exclusive primary inputs
    const exclusiveKeys = ['phase', 'totalField', 'localField'];
    const activeKey = exclusiveKeys.find(k => buckets[k].length > 0) || null;

    let html = '<div class="file-triage">';

    // --- Magnitude bucket (always shown) ---
    html += this._renderBucket('magnitude', allBucketDefs.magnitude, buckets.magnitude, {
      sublabel: buckets.magnitude.length > 0
        ? `${buckets.magnitude.length} file${buckets.magnitude.length > 1 ? 's' : ''}`
        : 'optional',
      eyeSvg, gripSvg, deleteSvg
    });

    // --- Exclusive primary inputs group ---
    html += '<div class="file-triage-exclusive-group">';
    html += '<div class="file-triage-group-label">Primary input <span class="file-triage-group-hint">choose one</span></div>';

    for (let i = 0; i < exclusiveKeys.length; i++) {
      const key = exclusiveKeys[i];
      const def = allBucketDefs[key];
      const items = buckets[key];
      const isActive = key === activeKey;
      const isInactive = activeKey && !isActive;

      let sublabel;
      if (isActive) {
        sublabel = 'active';
      } else if (isInactive) {
        sublabel = 'or use instead';
      } else {
        // No primary input yet
        sublabel = key === 'phase' ? 'multi-echo' : 'single file';
      }

      html += this._renderBucket(key, def, items, {
        sublabel,
        inactive: isInactive,
        eyeSvg, gripSvg, deleteSvg
      });

      // "or" divider between exclusive buckets
      if (i < exclusiveKeys.length - 1) {
        html += '<div class="file-triage-or-divider"><span>or</span></div>';
      }
    }

    html += '</div>'; // close exclusive group

    // --- JSON and Extra buckets (only if they have files) ---
    if (buckets.json.length > 0) {
      html += this._renderBucket('json', allBucketDefs.json, buckets.json, {
        sublabel: 'echo times', eyeSvg, gripSvg, deleteSvg
      });
    }

    if (buckets.extra.length > 0) {
      html += this._renderBucket('extra', allBucketDefs.extra, buckets.extra, {
        sublabel: 'drag to assign', isExtra: true, eyeSvg, gripSvg, deleteSvg
      });
    }

    html += '</div>';
    container.innerHTML = html;

    this._setupTriageDragDrop(container);
    this._setupTriageClickHandlers(container);
  }

  /**
   * Render a single bucket's HTML (header + drop zone + cards).
   */
  _renderBucket(key, def, items, opts = {}) {
    const { sublabel, inactive, isExtra, eyeSvg, gripSvg, deleteSvg } = opts;
    const emptyClass = items.length === 0 ? ' empty' : '';
    const inactiveClass = inactive ? ' inactive' : '';

    let html = `<div class="file-triage-bucket ${key}${inactiveClass}">`;
    html += `<div class="file-triage-bucket-header">`;
    html += `<span class="file-triage-bucket-label">${def.label} <span class="file-triage-bucket-count">(${items.length})</span>`;
    if (sublabel) {
      html += ` <span class="file-triage-bucket-sublabel">${sublabel}</span>`;
    }
    html += `</span>`;
    html += `<div class="file-triage-bucket-actions">`;

    if (isExtra && items.length > 0) {
      html += `<button class="btn-clear-extras" data-action="clearExtra" title="Remove all uncategorized">Clear</button>`;
    }

    if (def.canPreview && items.length > 0) {
      html += `<button class="btn-icon btn-preview file-triage-preview-btn" data-category="${key}" title="Preview ${def.label}">`;
      html += `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
      html += `</button>`;
    }

    html += `</div></div>`; // close bucket-actions, bucket-header

    html += `<div class="file-triage-drop${emptyClass}" data-bucket="${key}">`;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const teLabel = item.echoTime != null ? `${item.echoTime.toFixed(1)} ms` : '';

      html += `<div class="file-triage-card" draggable="true" data-category="${key}" data-index="${i}">`;
      html += `<span class="file-triage-card-grip" aria-hidden="true">${gripSvg}</span>`;
      html += `<span class="file-triage-card-name" title="${item.name}">${item.name}</span>`;
      if (teLabel) {
        html += `<span class="file-triage-card-te">${teLabel}</span>`;
      }
      if (def.canPreview) {
        html += `<button class="file-triage-card-preview" data-category="${key}" data-index="${i}" title="Preview">${eyeSvg}</button>`;
      }
      html += `<button class="file-triage-card-delete" data-category="${key}" data-index="${i}" title="Remove file">${deleteSvg}</button>`;
      html += `</div>`;
    }

    html += `</div>`; // close file-triage-drop
    html += `</div>`; // close file-triage-bucket
    return html;
  }

  /**
   * Update validation messages below file triage.
   */
  _updateInputValidation() {
    const container = document.getElementById('inputValidation');
    if (!container) return;

    const buckets = this.fileIOController.buckets;
    const mode = this.fileIOController.getInputMode();
    const messages = [];

    if (mode === 'raw') {
      const magCount = buckets.magnitude.length;
      const phaseCount = buckets.phase.length;

      if (phaseCount === 0) {
        messages.push({ type: 'error', text: 'Phase data required — drop phase NIfTI files or DICOM folder' });
      }
      if (magCount > 0 && phaseCount > 0 && magCount !== phaseCount) {
        messages.push({ type: 'error', text: `Echo count mismatch: ${magCount} magnitude, ${phaseCount} phase` });
      }
      if (magCount === 0 && phaseCount > 0) {
        messages.push({ type: 'info', text: 'Magnitude is optional but recommended for masking and some algorithms' });
      }
    } else if (mode === 'totalField') {
      if (buckets.magnitude.length === 0) {
        messages.push({ type: 'info', text: 'Magnitude is optional but recommended for masking' });
      }
    } else if (mode === 'localField') {
      if (buckets.magnitude.length === 0) {
        messages.push({ type: 'info', text: 'Magnitude is optional but recommended for masking' });
      }
    }

    if (messages.length === 0) {
      container.style.display = 'none';
      container.innerHTML = '';
      return;
    }

    container.style.display = '';
    container.innerHTML = messages
      .map(m => `<div class="validation-msg ${m.type}">${m.text}</div>`)
      .join('');
  }

  // Callbacks from FileIOController
  _onMagnitudeFilesChanged(files) {
    // Clear prepared state when magnitude files change
    this.maskPrepSettings.prepared = false;
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Update all dependent sections
    this.updateMagnitudePrepSection();
    this.updateMaskInputSourceOptions();
    this.updatePrepareButtonState();
    this.updateMaskSectionState();
    this.updateSidebarDropdownVisibility(true);

    if (files.length > 0) {
      this.visualizeMagnitude();
    }
  }

  _onPhaseFilesChanged(files) {
    // Phase files changed — triage UI handles preview buttons
  }

  // ==================== DICOM Handling ====================

  /**
   * Callback when DICOM conversion and classification is complete.
   * Routes results directly into FileIOController buckets.
   */
  _onDicomConversionComplete(classified) {
    // Sort by echo before adding to buckets
    const sortByEcho = (a, b) => {
      if (a.echoTime != null && b.echoTime != null) return a.echoTime - b.echoTime;
      if (a.echoNumber != null && b.echoNumber != null) return a.echoNumber - b.echoNumber;
      return 0;
    };
    classified.magnitude.sort(sortByEcho);
    classified.phase.sort(sortByEcho);

    // Push directly to FileIOController buckets
    const magnitudeFileData = classified.magnitude.map(e => ({
      file: e.file, name: e.name, echoTime: e.echoTime, echoNumber: e.echoNumber
    }));
    const phaseFileData = classified.phase.map(e => ({
      file: e.file, name: e.name, echoTime: e.echoTime, echoNumber: e.echoNumber
    }));
    const jsonFiles = classified.jsonFiles || [];

    this.fileIOController.setFilesFromDicom(magnitudeFileData, phaseFileData, jsonFiles);

    // Add extras to the extra bucket
    if (classified.extras?.length > 0) {
      for (const item of classified.extras) {
        this.fileIOController.buckets.extra.push({
          file: item.file, name: item.name, echoTime: item.echoTime, echoNumber: item.echoNumber
        });
      }
    }

    // Populate field strength if available
    if (classified.fieldStrength != null) {
      const fieldInput = document.getElementById('magField');
      if (fieldInput) fieldInput.value = classified.fieldStrength;
    }

    // Trigger UI update
    this._onBucketsChanged();
  }

  /**
   * Set up HTML5 drag-and-drop for reordering within buckets and moving between buckets.
   */
  _setupTriageDragDrop(container) {
    let dragData = null;

    container.querySelectorAll('.file-triage-card').forEach(card => {
      card.addEventListener('dragstart', (e) => {
        dragData = {
          category: card.dataset.category,
          index: parseInt(card.dataset.index)
        };
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', '');
      });

      card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        dragData = null;
        container.querySelectorAll('.file-triage-drop.dragover')
          .forEach(el => el.classList.remove('dragover'));
        container.querySelectorAll('.file-triage-card.drag-above, .file-triage-card.drag-below')
          .forEach(el => el.classList.remove('drag-above', 'drag-below'));
      });
    });

    container.querySelectorAll('.file-triage-drop').forEach(dropZone => {
      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        dropZone.classList.add('dragover');

        // Show drop position indicator for within-bucket reordering
        if (dragData && dropZone.dataset.bucket === dragData.category) {
          const cards = [...dropZone.querySelectorAll('.file-triage-card:not(.dragging)')];
          cards.forEach(c => c.classList.remove('drag-above', 'drag-below'));

          const closestCard = this._getClosestCard(cards, e.clientY);
          if (closestCard.element) {
            closestCard.element.classList.add(closestCard.after ? 'drag-below' : 'drag-above');
          }
        }
      });

      dropZone.addEventListener('dragleave', (e) => {
        if (!dropZone.contains(e.relatedTarget)) {
          dropZone.classList.remove('dragover');
          dropZone.querySelectorAll('.file-triage-card.drag-above, .file-triage-card.drag-below')
            .forEach(el => el.classList.remove('drag-above', 'drag-below'));
        }
      });

      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        dropZone.querySelectorAll('.file-triage-card.drag-above, .file-triage-card.drag-below')
          .forEach(el => el.classList.remove('drag-above', 'drag-below'));
        if (!dragData) return;

        const targetBucket = dropZone.dataset.bucket;
        const sourceBucket = dragData.category;

        if (targetBucket === sourceBucket) {
          // Reorder within the same bucket
          const cards = [...dropZone.querySelectorAll('.file-triage-card:not(.dragging)')];
          const closestCard = this._getClosestCard(cards, e.clientY);
          let toIndex;
          if (!closestCard.element) {
            toIndex = this.fileIOController.buckets[targetBucket].length - 1;
          } else {
            toIndex = parseInt(closestCard.element.dataset.index);
            if (closestCard.after) toIndex++;
            // Adjust if dragging from before the target
            if (dragData.index < toIndex) toIndex--;
          }
          this.fileIOController.reorderFile(targetBucket, dragData.index, toIndex);
        } else {
          // Move between buckets (enforces constraints)
          this.fileIOController.moveFile(sourceBucket, dragData.index, targetBucket);
        }
        this._onBucketsChanged();
      });
    });
  }

  /**
   * Find the closest card to the cursor Y position for drop insertion.
   * Returns { element, after } where after=true means insert after the element.
   */
  _getClosestCard(cards, y) {
    let closest = { element: null, after: false, distance: Infinity };

    for (const card of cards) {
      const rect = card.getBoundingClientRect();
      const midY = rect.top + rect.height / 2;
      const dist = Math.abs(y - midY);

      if (dist < closest.distance) {
        closest = { element: card, after: y > midY, distance: dist };
      }
    }

    return closest;
  }

  /**
   * Set up click handlers for triage card previews, deletes, and clear button.
   */
  _setupTriageClickHandlers(container) {
    // Individual card preview
    container.querySelectorAll('.file-triage-card-preview').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const category = btn.dataset.category;
        const index = parseInt(btn.dataset.index);
        const item = this.fileIOController.buckets[category]?.[index];
        if (item?.file) {
          this.viewerController.loadAndVisualizeFile(item.file, item.name);
        }
      });
    });

    // Bucket-level preview (eye in header)
    container.querySelectorAll('.file-triage-preview-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const category = btn.dataset.category;
        if (category === 'magnitude') this.visualizeMagnitude();
        else if (category === 'phase') this.visualizePhase();
        else if (category === 'totalField') this.visualizeFieldMap('totalField');
        else if (category === 'localField') this.visualizeFieldMap('localField');
      });
    });

    // Clear extra/uncategorized
    container.querySelector('[data-action="clearExtra"]')?.addEventListener('click', () => {
      this.fileIOController.buckets.extra = [];
      this._onBucketsChanged();
    });

    // Per-card delete
    container.querySelectorAll('.file-triage-card-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const category = btn.dataset.category;
        const index = parseInt(btn.dataset.index);
        this.fileIOController.removeFile(category, index);
        this._onBucketsChanged();
      });
    });
  }

  _showDicomStatus(text) {
    const el = document.getElementById('dicomStatus');
    const textEl = document.getElementById('dicomStatusText');
    if (el) el.style.display = '';
    if (textEl) textEl.textContent = text;
  }

  _hideDicomStatus() {
    const el = document.getElementById('dicomStatus');
    if (el) el.style.display = 'none';
  }

  // ==================== dicompare Integration ====================

  /**
   * Callback when DICOM files are retained for validation.
   */
  async _onDicomFilesRetained(files) {
    await this.dicompareController.retainDicomFiles(files);
    const btn = document.getElementById('dicompareReportBtn');
    if (btn) {
      btn.disabled = files.length === 0;
    }
  }

  /**
   * Run dicompare validation and display results in modal.
   */
  async runDicompareReport() {
    if (!this.dicompareController.hasFiles()) {
      this.updateOutput('No DICOM files available for validation.');
      return;
    }

    const body = document.getElementById('dicompareModalBody');
    const footer = document.getElementById('dicompareModalFooter');

    // If results are already cached, just re-display them
    const cached = this.dicompareController.getCachedResults();
    if (cached) {
      this.dicompareModal.open();
      this.dicompareRenderer.render(body, cached);
      if (footer) footer.style.display = '';
      return;
    }

    // Open modal with loading state
    this.dicompareModal.open();
    if (body) {
      body.innerHTML = `
        <div class="dicompare-loading">
          <div class="dicompare-spinner"></div>
          <p class="dicompare-loading-text" id="dicompareLoadingText">Initializing Python runtime...</p>
          <div class="dicompare-progress-bar">
            <div class="dicompare-progress-fill" id="dicompareProgressFill"></div>
          </div>
        </div>
      `;
    }
    if (footer) footer.style.display = 'none';

    try {
      const result = await this.dicompareController.runValidation((progress) => {
        const textEl = document.getElementById('dicompareLoadingText');
        const fillEl = document.getElementById('dicompareProgressFill');
        if (textEl) textEl.textContent = progress.currentOperation;
        if (fillEl) fillEl.style.width = `${progress.percentage}%`;
      });

      // Render results
      this.dicompareRenderer.render(body, result);
      if (footer) footer.style.display = '';
    } catch (error) {
      if (body) {
        body.innerHTML = `
          <div class="dicompare-error">
            <p>Validation failed: ${error.message}</p>
          </div>
        `;
      }
      console.error('dicompare validation error:', error);
    }
  }

  /**
   * Print the dicompare report in a new window.
   */
  printDicompareReport() {
    if (!this.dicompareController.complianceResults) return;
    const html = this.dicompareRenderer.generatePrintHtml({
      acquisitions: this.dicompareController.acquisitions,
      complianceResults: this.dicompareController.complianceResults,
      schema: JSON.parse(this.dicompareController.schemaContent || '{}')
    });
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(html);
      printWindow.document.close();
      printWindow.focus();
    }
  }

  // ==================== Sidebar Pipeline Dropdowns ====================

  setupSidebarDropdownListeners() {
    const mappings = [
      { id: 'sidebarCombinedMethod', key: 'combined_method' },
      { id: 'sidebarUnwrapMethod', key: 'unwrapping_algorithm' },
      { id: 'sidebarBgRemovalMethod', key: 'bf_algorithm' },
      { id: 'sidebarDipoleMethod', key: 'dipole_inversion' }
    ];

    for (const { id, key } of mappings) {
      document.getElementById(id)?.addEventListener('change', (e) => {
        this.pipelineSettings[key] = e.target.value;
        this.updateSidebarDropdownVisibility();
        this.updateInputParamsVisibility();
        this.updateEchoInfo();
      });
    }

  }

  updateSidebarDropdownVisibility(autoCorrect = false) {
    const mode = this.fileIOController?.getInputMode() || 'raw';
    const isRaw = mode === 'raw';
    const isTotalField = mode === 'totalField';
    const combined = this.pipelineSettings?.combined_method || 'none';
    const isStandard = combined === 'none';

    // Phase unwrapping: raw mode + standard pipeline only
    const unwrapGroup = document.getElementById('sidebarUnwrapGroup');
    if (unwrapGroup) unwrapGroup.style.display = (isRaw && isStandard) ? '' : 'none';

    // Background removal: (raw + standard) or (totalField + standard)
    const bgGroup = document.getElementById('sidebarBgRemovalGroup');
    if (bgGroup) bgGroup.style.display = ((isRaw || isTotalField) && isStandard) ? '' : 'none';

    // Dipole inversion: standard pipeline, any mode
    const dipoleGroup = document.getElementById('sidebarDipoleGroup');
    if (dipoleGroup) dipoleGroup.style.display = isStandard ? '' : 'none';

    // Magnitude gating
    const hasMagnitude = this.fileIOController?.buckets?.magnitude?.length > 0
      || this.preparedMagnitudeData !== null;
    const noMag = !hasMagnitude;

    // Auto-correct to safe defaults when data changes
    const combinedSelect = document.getElementById('sidebarCombinedMethod');
    if (combinedSelect) {
      if (autoCorrect && noMag && combinedSelect.value === 'qsmart') {
        combinedSelect.value = 'none';
        this.pipelineSettings.combined_method = 'none';
      }
      this._showSidebarWarning('sidebarCombinedMethod', 'sidebarCombinedWarning',
        noMag && combinedSelect.value === 'qsmart',
        'Requires magnitude');
    }

    const dipoleSelect = document.getElementById('sidebarDipoleMethod');
    if (dipoleSelect) {
      if (autoCorrect && noMag && dipoleSelect.value === 'medi') {
        dipoleSelect.value = 'rts';
        this.pipelineSettings.dipole_inversion = 'rts';
      }
      this._showSidebarWarning('sidebarDipoleMethod', 'sidebarDipoleWarning',
        noMag && dipoleSelect.value === 'medi',
        'Requires magnitude');
    }
  }

  _showSidebarWarning(anchorId, warningId, show, message) {
    let warning = document.getElementById(warningId);
    const anchor = document.getElementById(anchorId);

    if (show) {
      if (!warning && anchor) {
        warning = document.createElement('div');
        warning.id = warningId;
        warning.className = 'validation-message error inline-warning';
        warning.innerHTML = '<span></span>';
        anchor.parentNode.insertBefore(warning, anchor.nextSibling);
      }
      if (warning) {
        warning.querySelector('span').textContent = message;
        warning.style.display = 'flex';
      }
    } else if (warning) {
      warning.style.display = 'none';
    }
  }

  syncSidebarFromSettings() {
    const s = this.pipelineSettings;
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.value = val;
    };
    set('sidebarCombinedMethod', s?.combined_method || 'none');
    set('sidebarUnwrapMethod', s?.unwrapping_algorithm || 'romeo');
    set('sidebarBgRemovalMethod', s?.bf_algorithm || 'vsharp');
    set('sidebarDipoleMethod', s?.dipole_inversion || 'tv');
    this.updateSidebarDropdownVisibility();
  }

  // switchInputMode removed — input mode is now derived from bucket contents

  updateInputParamsVisibility() {
    const mode = this.fileIOController.getInputMode();
    const isRaw = mode === 'raw';
    const units = this.fileIOController.getFieldMapUnits();
    const combined_method = this.pipelineSettings?.combined_method || 'none';
    // Field strength needed for: raw mode, Hz/rad_s units, or TGV/QSMART (internal scaling)
    const needsFieldStrength = isRaw || units !== 'ppm' || combined_method !== 'none';

    // Show/hide raw-mode-only parameters
    const echoTimesGroup = document.getElementById('echoTimesGroup');
    if (echoTimesGroup) echoTimesGroup.style.display = isRaw ? '' : 'none';

    // Show/hide field map units (only for field map modes)
    const unitsGroup = document.getElementById('fieldMapUnitsGroup');
    if (unitsGroup) unitsGroup.style.display = isRaw ? 'none' : '';

    // Show/hide field strength
    const fieldGroup = document.getElementById('fieldStrengthGroup');
    if (fieldGroup) fieldGroup.style.display = needsFieldStrength ? '' : 'none';
  }

  updateMagnitudePrepSection() {
    const section = document.getElementById('maskSection');
    if (!section) return;

    const hasMag = this.fileIOController.buckets.magnitude.length > 0;
    const hasPhase = this.fileIOController.buckets.phase.length > 0;
    const source = this.maskPrepSettings.source;
    const isCustom = source === 'custom';
    const canPrepare = isCustom || (source === 'phase_quality' ? hasPhase : hasMag);

    // Show/hide inline warning instead of greying out
    let warning = document.getElementById('magnitudePrepWarning');
    if (!canPrepare) {
      const msg = source === 'phase_quality' ? 'Requires phase data' : 'Requires magnitude data';
      if (!warning) {
        warning = document.createElement('div');
        warning.id = 'magnitudePrepWarning';
        warning.className = 'validation-message error inline-warning';
        warning.innerHTML = `<span>${msg}</span>`;
        const content = section.querySelector('.section-content');
        if (content) content.prepend(warning);
      } else {
        warning.querySelector('span').textContent = msg;
      }
      warning.style.display = 'flex';
    } else if (warning) {
      warning.style.display = 'none';
    }

    // Enable/disable Prepare button based on data availability for the selected source
    const prepareBtn = document.getElementById('prepareMaskInput');
    if (prepareBtn) prepareBtn.disabled = !canPrepare;
  }

  updateMaskSectionState() {
    const section = document.getElementById('maskSection');
    if (!section) return;

    const hasMaskFile = this.fileIOController.hasMask();
    const hasPrepared = this.maskPrepSettings.prepared;
    const isCustom = this.maskPrepSettings.source === 'custom';

    const generateButtons = document.getElementById('maskGenerateButtons');
    const thresholdModeButtons = document.getElementById('thresholdModeButtons');
    const thresholdSliderGroup = document.getElementById('thresholdSliderGroup');
    const maskOps = document.getElementById('maskOperations');

    // Hide generation controls entirely for custom mask mode
    if (isCustom) {
      if (generateButtons) generateButtons.style.display = 'none';
      if (thresholdModeButtons) thresholdModeButtons.style.display = 'none';
      if (thresholdSliderGroup) thresholdSliderGroup.style.display = 'none';
      if (maskOps) maskOps.style.display = 'none';
      const maskFileNote = document.getElementById('maskFileUploadedNote');
      if (maskFileNote) maskFileNote.style.display = 'none';
      return;
    }

    if (generateButtons) generateButtons.style.display = '';

    // When mask file uploaded: disable Threshold/BET generation controls + show info note
    const maskFileNote = document.getElementById('maskFileUploadedNote');
    if (hasMaskFile) {
      if (generateButtons) generateButtons.style.opacity = '0.5';
      document.getElementById('previewMask')?.setAttribute('disabled', '');
      document.getElementById('runBET')?.setAttribute('disabled', '');
      document.getElementById('maskThreshold')?.setAttribute('disabled', '');
      if (maskOps) maskOps.style.display = 'none';
      // Show info note
      if (!maskFileNote) {
        const note = document.createElement('div');
        note.id = 'maskFileUploadedNote';
        note.className = 'validation-message info inline-warning';
        note.innerHTML = '<span>Using uploaded mask file. Remove it to use generated masks.</span>';
        if (generateButtons) generateButtons.parentNode.insertBefore(note, generateButtons.nextSibling);
      } else {
        maskFileNote.style.display = 'flex';
      }
    } else {
      if (maskFileNote) maskFileNote.style.display = 'none';
      if (hasPrepared) {
        // Magnitude prepared: enable generation controls
        if (generateButtons) generateButtons.style.opacity = '1';
        document.getElementById('previewMask')?.removeAttribute('disabled');
        document.getElementById('runBET')?.removeAttribute('disabled');
      }
    }
  }

  updateMaskInputSourceOptions() {
    const magCount = this.fileIOController.buckets.magnitude.length;

    const select = document.getElementById('maskInputSource');
    if (!select) return;

    const combinedOption = select.querySelector('option[value="combined"]');
    if (combinedOption) {
      combinedOption.disabled = magCount <= 1;
      combinedOption.title = magCount <= 1 ? 'Requires 2+ magnitude files for RSS combination' : '';
      // Force to first_echo if combined was selected but only 1 file
      if (magCount <= 1 && select.value === 'combined') {
        select.value = 'first_echo';
        this.maskPrepSettings.source = 'first_echo';
      }
    }
  }

  updatePipelineSectionForMode(mode) {
    // Update the run button label based on input mode
    const runButton = document.getElementById('runPipelineSidebar');
    if (runButton) {
      const label = runButton.querySelector('span');
      if (label) {
        const labels = {
          raw: 'Start QSM',
          totalField: 'Start QSM',
          localField: 'Start QSM'
        };
        label.textContent = labels[mode] || 'Start QSM';
      }
    }
  }

  async visualizeFieldMap(type) {
    const file = type === 'totalField'
      ? this.fileIOController.getTotalFieldFile()
      : this.fileIOController.getLocalFieldFile();
    if (!file) return;

    const label = type === 'totalField' ? 'Total Field Map' : 'Local Field Map';
    await this.loadAndVisualizeFile(file, label);
    this.hideEchoNavigation();
  }

  async visualizeFieldMapMagnitude(type) {
    const file = this.fileIOController.getFieldMapMagnitudeFile();
    if (!file) return;

    await this.loadAndVisualizeFile(file, 'Magnitude');
    this.hideEchoNavigation();
  }

  async visualizeMaskFile() {
    const file = this.fileIOController.getMaskFile();
    if (!file) return;

    await this.loadAndVisualizeFile(file, 'Mask');
    this.hideEchoNavigation();
  }

  // Update run button state based on file/mask state (mode-aware)
  updateEchoInfo() {
    const mode = this.fileIOController?.getInputMode() || 'raw';
    const isValid = this.fileIOController?.hasValidData() || false;
    const combined_method = this.pipelineSettings?.combined_method || 'none';
    let canRun = false;

    switch (mode) {
      case 'raw': {
        const hasEchoTimes = this.fileIOController?.hasEchoTimes() || false;
        const hasMask = this.currentMaskData !== null;
        canRun = isValid && hasEchoTimes && hasMask;
        break;
      }
      case 'totalField':
      case 'localField': {
        const units = this.fileIOController.getFieldMapUnits();
        // Field strength needed for Hz/rad_s, and for TGV/QSMART internal scaling
        const needsFieldStrength = units !== 'ppm' || combined_method !== 'none';
        const hasFieldStrength = !needsFieldStrength || (this.fileIOController.getFieldStrength() > 0);
        // Mask can come from: UI editing, mask file upload, or magnitude (for threshold generation)
        const hasMaskSource = this.currentMaskData !== null
          || this.fileIOController.hasMask()
          || this.fileIOController.hasFieldMapMagnitude()
          || this.preparedMagnitudeData !== null;
        // QSMART and MEDI require magnitude
        const dipoleMethod = this.pipelineSettings?.dipole_inversion || 'rts';
        const needsMagnitude = combined_method === 'qsmart' || dipoleMethod === 'medi';
        const hasMagnitude = this.fileIOController.hasFieldMapMagnitude()
          || this.preparedMagnitudeData !== null;
        const algorithmOk = !needsMagnitude || hasMagnitude;
        canRun = isValid && hasFieldStrength && hasMaskSource && algorithmOk;
        break;
      }
    }

    const runButton = document.getElementById('runPipelineSidebar');
    if (runButton) {
      runButton.disabled = !canRun || this.pipelineRunning;
    }
  }

  // Delegate to FileIOController
  getEchoTimesFromInputs() {
    return this.fileIOController?.getEchoTimesFromInputs() || [];
  }

  // Visualization methods - delegate to ViewerController
  async visualizeMagnitude() {
    await this.viewerController.visualizeMagnitude();
    this.syncViewerState();
  }

  async visualizePhase() {
    await this.viewerController.visualizePhase();
    this.syncViewerState();
  }

  navigateEcho(direction) {
    this.viewerController.navigateEcho(direction);
    this.syncViewerState();
  }

  async visualizeCurrentEcho() {
    await this.viewerController.visualizeCurrentEcho();
    this.syncViewerState();
  }

  updateEchoNavigation() {
    this.viewerController.updateEchoNavigation();
  }

  hideEchoNavigation() {
    this.viewerController.hideEchoNavigation();
    this.currentViewType = null;
  }

  async loadAndVisualizeFile(file, description) {
    await this.viewerController.loadAndVisualizeFile(file, description);
    this.currentFile = this.viewerController.getCurrentFile();
  }

  updateDataUnits(description) {
    this.viewerController.updateDataUnits(description);
  }

  // Sync viewer state from controller to app
  syncViewerState() {
    this.currentEchoIndex = this.viewerController.getCurrentEchoIndex();
    this.currentViewType = this.viewerController.getCurrentViewType();
    this.currentFile = this.viewerController.getCurrentFile();
  }

  /**
   * Update the Prepare button state based on current settings
   */
  updatePrepareButtonState() {
    const btn = document.getElementById('prepareMaskInput');
    const source = this.maskPrepSettings.source;
    const canPrepare = source === 'phase_quality'
      ? this.fileIOController?.buckets?.phase?.length > 0
      : this.fileIOController?.buckets?.magnitude?.length > 0;

    if (btn) {
      btn.disabled = !canPrepare;
    }

    // Enable/disable masking controls based on prepared state
    this.updateMaskingControlsState();
  }

  /**
   * Enable or disable masking controls based on whether Prepare has been run
   */
  updateMaskingControlsState() {
    const prepared = this.maskPrepSettings.prepared;
    const hasMaskFile = this.fileIOController?.hasMask() || false;

    // Threshold/BET enabled when prepared AND no mask file uploaded directly
    const canGenerate = prepared && !hasMaskFile;

    // Preview Mask (Threshold) button
    const previewBtn = document.getElementById('previewMask');
    if (previewBtn) previewBtn.disabled = !canGenerate;

    // BET button
    const betBtn = document.getElementById('runBET');
    if (betBtn) betBtn.disabled = !canGenerate;

    // Threshold slider and auto-threshold button:
    // Only enabled when Threshold method is active (not BET)
    // This is controlled separately by setThresholdSliderEnabled()
    // On initial prepare, keep them disabled until user clicks Threshold

    // Morphological operations panel - show/hide based on state
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) {
      // Only show if prepared AND we have a mask (from either Threshold or BET)
      opsPanel.style.display = (prepared && this.currentMaskData) ? 'block' : 'none';
    }
  }

  /**
   * Enable or disable the threshold slider and auto-threshold button
   */
  setThresholdSliderEnabled(enabled) {
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider) thresholdSlider.disabled = !enabled;

    const autoThresholdBtn = document.getElementById('autoThreshold');
    if (autoThresholdBtn) autoThresholdBtn.disabled = !enabled;
  }

  /**
   * Prepare mask input by combining echoes and/or applying bias correction
   * Delegates to MaskController
   */
  async prepareMaskInput() {
    // Get magnitude and phase files from unified buckets
    const magnitudeFiles = this.fileIOController.buckets.magnitude;
    const phaseFiles = this.fileIOController.buckets.phase;

    await this.maskController.prepareMaskInput({
      magnitudeFiles: magnitudeFiles,
      phaseFiles: phaseFiles,
      echoTimes: this.getEchoTimesFromInputs(),
      maskPrepSettings: this.maskPrepSettings,
      onComplete: () => {
        // Sync state from controller to app
        this.magnitudeData = this.maskController.magnitudeData;
        this.magnitudeMax = this.maskController.magnitudeMax;
        this.preparedMagnitudeData = this.maskController.preparedMagnitudeData;
        this.preparedMagnitudeMax = this.maskController.preparedMagnitudeMax;
        this.magnitudeFileBytes = this.maskController.magnitudeFileBytes;
        this.magnitudeVolume = this.maskController.magnitudeVolume;

        this.maskPrepSettings.prepared = true;
        this.updatePrepareButtonState();
        this.updateMaskSectionState();
        this.hideEchoNavigation();
        this.showStageButtons();
        this.addStageButton('preparedMagnitude', 'Masking input');
      }
    });
  }

  /**
   * Display the prepared magnitude data as the base volume
   * Delegates to MaskController
   */
  async displayPreparedMagnitude() {
    await this.maskController.displayPreparedMagnitude();
    this.magnitudeVolume = this.maskController.magnitudeVolume;
    this.updateDownloadVolumeButton();
  }

  // Create NIfTI header from NiiVue volume - delegates to imported module
  createNiftiHeaderFromVolume(vol) {
    return createNiftiHeaderFromVolume(vol);
  }

  /**
   * Read NIfTI header from a file without displaying it
   * Delegates to MaskController
   */
  async readNiftiHeader(file) {
    return this.maskController.readNiftiHeader(file);
  }

  /**
   * Read NIfTI image data from a file without displaying it
   * Delegates to MaskController
   */
  async readNiftiData(file) {
    return this.maskController.readNiftiData(file);
  }

  /**
   * Combine multiple magnitude echoes using Root Sum of Squares (RSS)
   * Delegates to MaskController
   */
  async combineMagnitudeRSS() {
    return this.maskController.combineMagnitudeRSS(this.fileIOController.buckets.magnitude);
  }

  /**
   * Apply bias field correction to magnitude data
   * Delegates to MaskController
   */
  async applyBiasCorrection(magnitudeData) {
    return this.maskController.applyBiasCorrection(magnitudeData);
  }

  /**
   * Preview mask based on threshold
   * Delegates to MaskController
   */
  async previewMask() {
    // Sync threshold to controller before previewing
    this.maskController.setMaskThreshold(this.maskThreshold);

    await this.maskController.previewMask(this.maskPrepSettings);

    // Sync state from controller
    this.currentMaskData = this.maskController.currentMaskData;
    this.originalMaskData = this.maskController.originalMaskData;
    this.maskDims = this.maskController.maskDims;
    this.voxelSize = this.maskController.voxelSize;
    this.applyVoxelDefaults();

    // Show morphological operations panel
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'block';

    // Add mask to Results section
    this.showStageButtons();
    this.addStageButton('mask', 'Brain Mask');

    // Update run button state
    this.updateEchoInfo();
  }

  /**
   * Update mask preview when threshold changes
   * Delegates to MaskController
   */
  async updateMaskPreview() {
    // Sync threshold to controller
    this.maskController.setMaskThreshold(this.maskThreshold);

    await this.maskController.updateMaskPreview();

    // Sync state from controller
    this.currentMaskData = this.maskController.currentMaskData;
    this.originalMaskData = this.maskController.originalMaskData;
    this.maskDims = this.maskController.maskDims;
    this.voxelSize = this.maskController.voxelSize;

    // Show morphological operations panel
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'block';

    // Add mask to Results section
    this.showStageButtons();
    this.addStageButton('mask', 'Brain Mask');

    // Update run button state
    this.updateEchoInfo();
  }

  /**
   * Display the current mask as an overlay
   * Delegates to MaskController
   */
  async displayCurrentMask() {
    // Sync mask data to controller if it was modified locally
    if (this.currentMaskData !== this.maskController.currentMaskData) {
      this.maskController.currentMaskData = this.currentMaskData;
    }
    await this.maskController.displayCurrentMask();
  }

  /**
   * Update the opacity of all overlay volumes
   */
  updateOverlayOpacity(opacity) {
    if (this.nv.volumes.length <= 1) return;

    // Update all overlays (volumes after the first one)
    for (let i = 1; i < this.nv.volumes.length; i++) {
      this.nv.setOpacity(i, opacity);
    }
    this.nv.updateGLVolume();
  }

  /**
   * Show or hide the overlay opacity control
   */
  showOverlayControl(show) {
    const control = document.getElementById('overlayOpacityControl');
    if (control) {
      control.style.display = show ? 'flex' : 'none';
    }
  }

  /**
   * Update the download volume button state
   */
  updateDownloadVolumeButton() {
    const btn = document.getElementById('downloadCurrentVolume');
    if (btn) {
      btn.disabled = !this.nv.volumes || this.nv.volumes.length === 0;
    }
  }

  // 3D morphological erosion - delegates to MaskController
  // Append an erode/dilate op to the mask history, collapsing successive same-type ops into a
  // single iteration count (so three erode clicks record `erode:3`, not `erode:1,erode:1,erode:1`).
  // This keeps the generated `--mask ...` string and the methods prose concise and correct.
  _pushMaskOp(type) {
    const h = this.maskOpsHistory;
    const last = h[h.length - 1];
    const m = last && /^(erode|dilate):(\d+)$/.exec(last);
    if (m && m[1] === type) {
      h[h.length - 1] = `${type}:${parseInt(m[2], 10) + 1}`;
    } else {
      h.push(`${type}:1`);
    }
  }

  erodeMask3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.erodeMask3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // 3D morphological dilation - delegates to MaskController
  dilateMask3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.dilateMask3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // Fill holes in 3D mask - delegates to MaskController
  fillHoles3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.fillHoles3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // Clear mask completely - delegates to MaskController
  async clearMask() {
    await this.maskController.clearMask();

    // Sync state
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Hide threshold slider
    const sliderGroup = document.getElementById('thresholdSliderGroup');
    if (sliderGroup) sliderGroup.style.display = 'none';

    // Update run button state (mask no longer available)
    this.updateEchoInfo();
  }

  // Toggle drawing mode on/off - delegates to MaskController
  async toggleDrawingMode() {
    // Sync mask data to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;
    this.maskController.brushSize = this.brushSize;

    await this.maskController.toggleDrawingMode();

    // Sync state back
    this.drawingEnabled = this.maskController.drawingEnabled;
    this.brushMode = this.maskController.brushMode;
    this.savedCrosshairWidth = this.maskController.savedCrosshairWidth;
  }

  // Set brush mode (add or remove) - delegates to MaskController
  setBrushMode(mode) {
    this.maskController.setBrushMode(mode);
    this.brushMode = this.maskController.brushMode;
  }

  // Set brush size - delegates to MaskController
  setBrushSize(size) {
    this.brushSize = size;
    this.maskController.setBrushSize(size);
  }

  // Toggle 3D brush - delegates to MaskController
  setBrush3D(enabled) {
    this.maskController.setBrush3D(enabled);
  }

  // Set brush shape (square or circle) - delegates to MaskController
  setBrushShape(shape) {
    this.maskController.setBrushShape(shape);
  }

  // Apply the drawing to the current mask - delegates to MaskController
  async applyDrawingToMask() {
    // Sync mask data to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    await this.maskController.applyDrawingToMask();

    // Sync state back
    this.currentMaskData = this.maskController.currentMaskData;
    this.drawingEnabled = this.maskController.drawingEnabled;
    this.brushMode = this.maskController.brushMode;

    // Update run button state
    this.updateEchoInfo();
  }

  // Create mask NIfTI using source header as template - delegates to imported module
  createMaskNifti(maskData) {
    return createMaskNifti(maskData, this.magnitudeFileBytes);
  }

  async runRomeoQSM() {
    const mode = this.fileIOController.getInputMode();

    if (mode === 'raw') {
      await this._runRawPipeline();
    } else if (mode === 'totalField') {
      await this._runTotalFieldPipeline();
    } else if (mode === 'localField') {
      await this._runLocalFieldPipeline();
    }
  }

  async _runRawPipeline() {
    // Validation
    const magCount = this.fileIOController.buckets.magnitude.length;
    const phaseCount = this.fileIOController.buckets.phase.length;
    const echoTimes = this.getEchoTimesFromInputs();
    const echoTimeCount = echoTimes.length;

    if (magCount === 0 || phaseCount === 0) {
      this.updateOutput("Please upload both magnitude and phase files");
      return;
    }

    if (magCount !== phaseCount) {
      this.updateOutput(`File count mismatch: ${magCount} magnitude, ${phaseCount} phase`);
      return;
    }

    if (echoTimeCount === 0) {
      this.updateOutput("Please enter echo times");
      return;
    }

    // Get parameters
    const magField = parseFloat(document.getElementById('magField').value);

    if (!magField || magField <= 0) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    try {
      // Read file buffers
      const magnitudeBuffers = [];
      const phaseBuffers = [];

      for (let i = 0; i < magCount; i++) {
        const magFile = this.fileIOController.buckets.magnitude[i]?.file;
        const phaseFile = this.fileIOController.buckets.phase[i]?.file;

        if (magFile && phaseFile) {
          magnitudeBuffers.push(await magFile.arrayBuffer());
          phaseBuffers.push(await phaseFile.arrayBuffer());
        }
      }

      // Prepare custom mask if available
      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using edited mask");
      }

      // Determine which stages can be skipped based on settings changes
      const skipStages = this.pipelineExecutor.determineSkipStages(this.pipelineSettings);
      if (skipStages.skipUnwrap) {
        this.updateOutput("Reusing cached unwrapped phase data");
      }
      if (skipStages.skipBgRemoval) {
        this.updateOutput("Reusing cached background-removed data");
      }

      // Show phase image at the start
      await this.visualizePhase();

      // Include prepared magnitude if available (for MEDI gradient weighting and threshold mask)
      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      // Run pipeline via executor
      const started = await this.pipelineExecutor.run({
        inputMode: 'raw',
        magnitudeBuffers,
        phaseBuffers,
        echoTimes,
        magField,
        maskThreshold: this.maskThreshold,
        customMaskBuffer,
        preparedMagnitude,
        pipelineSettings: this.pipelineSettings,
        skipStages
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  async _runTotalFieldPipeline() {
    const totalFieldFile = this.fileIOController.getTotalFieldFile();
    if (!totalFieldFile) {
      this.updateOutput("Please upload a total field map file");
      return;
    }

    const units = this.fileIOController.getFieldMapUnits();
    const combined_method = this.pipelineSettings?.combined_method || 'none';
    // Field strength needed for Hz/rad_s conversion, and for TGV/QSMART internal scaling
    const needsFieldStrength = units !== 'ppm' || combined_method !== 'none';
    const magField = needsFieldStrength ? parseFloat(document.getElementById('magField').value) : null;

    if (needsFieldStrength && (!magField || magField <= 0)) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    // QSMART and MEDI require magnitude
    const dipoleMethod = this.pipelineSettings?.dipole_inversion || 'rts';
    if ((combined_method === 'qsmart' || dipoleMethod === 'medi')
        && !this.fileIOController.hasFieldMapMagnitude() && !this.preparedMagnitudeData) {
      const method = combined_method === 'qsmart' ? 'QSMART' : 'MEDI';
      this.updateOutput(`${method} requires a magnitude image`);
      return;
    }

    try {
      const totalFieldBuffer = await totalFieldFile.arrayBuffer();

      // Extract voxel size from NIfTI header for pipeline defaults
      if (!isGzipped(new Uint8Array(totalFieldBuffer)) && totalFieldBuffer.byteLength >= 352) {
        const headerInfo = parseNiftiHeader(totalFieldBuffer.slice(0, 352));
        this.voxelSize = headerInfo.voxelSize;
        this.maskDims = [headerInfo.nx, headerInfo.ny, headerInfo.nz];
        this.applyVoxelDefaults();
      }

      // Optional magnitude for masking/MEDI/QSMART vasculature
      const magFile = this.fileIOController.getFieldMapMagnitudeFile();
      const magnitudeBuffer = magFile ? await magFile.arrayBuffer() : null;

      // Mask: from centralized upload, from UI editing, or will be generated from magnitude
      const maskFile = this.fileIOController.getMaskFile();
      let maskBuffer = maskFile ? await maskFile.arrayBuffer() : null;

      // Use custom edited mask if available
      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using edited mask");
      }

      // Preview the field map
      await this.visualizeFieldMap('totalField');

      const started = await this.pipelineExecutor.run({
        inputMode: 'totalField',
        totalFieldBuffer,
        fieldMapUnits: units,
        magnitudeBuffer,
        maskBuffer,
        customMaskBuffer,
        magField,
        maskThreshold: this.maskThreshold,
        preparedMagnitude: this.preparedMagnitudeData ? Array.from(this.preparedMagnitudeData) : null,
        pipelineSettings: this.pipelineSettings
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  async _runLocalFieldPipeline() {
    const localFieldFile = this.fileIOController.getLocalFieldFile();
    if (!localFieldFile) {
      this.updateOutput("Please upload a local field map file");
      return;
    }

    const units = this.fileIOController.getFieldMapUnits();
    const combined_method = this.pipelineSettings?.combined_method || 'none';
    const needsFieldStrength = units !== 'ppm' || combined_method !== 'none';
    const magField = needsFieldStrength ? parseFloat(document.getElementById('magField').value) : null;

    if (needsFieldStrength && (!magField || magField <= 0)) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    // QSMART and MEDI require magnitude
    const dipoleMethod = this.pipelineSettings?.dipole_inversion || 'rts';
    if ((combined_method === 'qsmart' || dipoleMethod === 'medi')
        && !this.fileIOController.hasFieldMapMagnitude() && !this.preparedMagnitudeData) {
      const method = combined_method === 'qsmart' ? 'QSMART' : 'MEDI';
      this.updateOutput(`${method} requires a magnitude image`);
      return;
    }

    try {
      const localFieldBuffer = await localFieldFile.arrayBuffer();

      // Extract voxel size from NIfTI header for pipeline defaults
      if (!isGzipped(new Uint8Array(localFieldBuffer)) && localFieldBuffer.byteLength >= 352) {
        const headerInfo = parseNiftiHeader(localFieldBuffer.slice(0, 352));
        this.voxelSize = headerInfo.voxelSize;
        this.maskDims = [headerInfo.nx, headerInfo.ny, headerInfo.nz];
        this.applyVoxelDefaults();
      }

      // Optional magnitude for MEDI/QSMART vasculature
      const magFile = this.fileIOController.getFieldMapMagnitudeFile();
      const magnitudeBuffer = magFile ? await magFile.arrayBuffer() : null;

      // Mask: from centralized upload or from UI editing
      const maskFile = this.fileIOController.getMaskFile();
      let maskBuffer = maskFile ? await maskFile.arrayBuffer() : null;

      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using edited mask");
      }

      if (!maskBuffer && !customMaskBuffer && combined_method === 'none') {
        this.updateOutput("Please provide a mask file or create one from magnitude");
        return;
      }

      // Preview the field map
      await this.visualizeFieldMap('localField');

      const started = await this.pipelineExecutor.run({
        inputMode: 'localField',
        localFieldBuffer,
        fieldMapUnits: units,
        magnitudeBuffer,
        maskBuffer,
        customMaskBuffer,
        magField,
        maskThreshold: this.maskThreshold,
        preparedMagnitude: this.preparedMagnitudeData ? Array.from(this.preparedMagnitudeData) : null,
        pipelineSettings: this.pipelineSettings
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  cancelPipeline() {
    this.pipelineExecutor?.cancel();
    document.getElementById('cancelPipeline').disabled = true;
    this.updateEchoInfo();
  }

  showStageButtons() {
    const resultsSection = document.getElementById('stage-buttons');
    resultsSection.classList.remove('hidden');
    // Expand results section if collapsed (don't close if already open)
    if (resultsSection.classList.contains('collapsed')) {
      resultsSection.classList.remove('collapsed');
    }
  }

  // Create or update a stage button dynamically
  addStageButton(stage, description) {
    const container = document.getElementById('dynamicStageButtons');
    if (!container) return;

    // Check if button already exists
    const existingItem = document.getElementById(`stage-item-${stage}`);
    if (existingItem) {
      // Already exists, just make sure it's enabled
      const showBtn = existingItem.querySelector('.stage-tab');
      const downloadBtn = existingItem.querySelector('.stage-download');
      if (showBtn) showBtn.disabled = false;
      if (downloadBtn) downloadBtn.disabled = false;
      return;
    }

    // Track stage order
    if (!this.stageOrder.includes(stage)) {
      this.stageOrder.push(stage);
    }

    // Create display name from stage ID or description
    const displayName = this.getStageDisplayName(stage, description);

    // Create the stage item
    const stageItem = document.createElement('div');
    stageItem.className = 'stage-item';
    stageItem.id = `stage-item-${stage}`;

    // Show button
    const showBtn = document.createElement('button');
    showBtn.className = 'btn btn-secondary btn-sm stage-tab';
    showBtn.textContent = displayName;
    showBtn.title = description || stage;
    showBtn.addEventListener('click', () => this.showStage(stage));

    // Download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-secondary btn-sm btn-icon stage-download';
    downloadBtn.title = `Download ${displayName} as NIfTI`;
    downloadBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
    downloadBtn.addEventListener('click', () => this.downloadStage(stage));

    stageItem.appendChild(showBtn);
    stageItem.appendChild(downloadBtn);
    container.appendChild(stageItem);
  }

  /**
   * Download a specific stage result as NIfTI
   */
  downloadStage(stage) {
    // Pipeline results stored as File objects
    if (this.results[stage]?.file) {
      const file = this.results[stage].file;
      const url = URL.createObjectURL(file);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name || `${stage}.nii`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.updateOutput(`Downloaded: ${a.download}`);
      return;
    }

    // Mask (local data)
    const headerBytes = this.magnitudeFileBytes || this.maskController.magnitudeFileBytes;
    if (stage === 'mask') {
      const maskData = this.currentMaskData || this.maskController.currentMaskData;
      if (maskData && headerBytes) {
        const nifti = createMaskNifti(maskData, headerBytes);
        this._downloadBuffer(nifti, 'brain_mask.nii');
        return;
      }
    }

    // Prepared magnitude (local data)
    if (stage === 'preparedMagnitude') {
      const prepData = this.preparedMagnitudeData || this.maskController.preparedMagnitudeData;
      if (prepData && headerBytes) {
        const nifti = createFloat64Nifti(prepData, headerBytes);
        this._downloadBuffer(nifti, 'masking_input.nii');
        return;
      }
    }

    this.updateOutput(`No data available to download for ${stage}`);
  }

  _downloadBuffer(buffer, filename) {
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    this.updateOutput(`Downloaded: ${filename}`);
  }

  // Get a user-friendly display name for a stage
  getStageDisplayName(stage, description) {
    const nameMap = window.QSMConfig.STAGE_DISPLAY_NAMES;

    // Use mapped name, or extract short name from description, or use stage ID
    if (nameMap[stage]) {
      return nameMap[stage];
    }

    // Try to extract a short name from description (first 2 words)
    if (description) {
      const words = description.split(' ').slice(0, 2).join(' ');
      if (words.length <= 15) return words;
    }

    // Fallback to stage ID (camelCase to Title Case)
    return stage.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase()).trim();
  }

  // Clear all dynamic stage buttons (called at pipeline start)
  clearStageButtons() {
    const container = document.getElementById('dynamicStageButtons');
    if (container) {
      container.innerHTML = '';
    }
  }

  // Clear all cached results (internal use)
  clearResults() {
    this.pipelineExecutor?.clearResults();
  }

  // Clear all results including prepared magnitude and mask (user-triggered)
  clearAllResults() {
    // Clear pipeline results
    this.clearResults();
    this.clearStageButtons();

    // Clear prepared magnitude
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;
    this.maskPrepSettings.prepared = false;

    // Clear mask
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Hide morphological operations panel and threshold slider
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'none';
    const sliderGroup = document.getElementById('thresholdSliderGroup');
    if (sliderGroup) sliderGroup.style.display = 'none';

    // Hide Results section
    const resultsSection = document.getElementById('stage-buttons');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    // Update button states
    this.updatePrepareButtonState();
    this.updateEchoInfo();

    this.updateOutput("Results cleared");
  }

  updateDownloadButtons() {
    // Legacy method - now handled by enableStageButtons
  }

  async showStage(stage) {
    try {
      // For magnitude and phase, use the multi-echo viewer with echo navigation
      if (stage === 'magnitude' || stage === 'phase') {
        this.currentViewType = stage;
        this.currentEchoIndex = 0;
        await this.visualizeCurrentEcho();
        this.updateEchoNavigation();
        return;
      }

      // For single 3D volume stages, hide echo navigation
      this.hideEchoNavigation();

      // Handle prepared magnitude (local data, not from pipeline)
      if (stage === 'preparedMagnitude') {
        if (this.preparedMagnitudeData || this.maskController.preparedMagnitudeData) {
          if (!this.preparedMagnitudeData) {
            this.preparedMagnitudeData = this.maskController.preparedMagnitudeData;
          }
          await this.displayPreparedMagnitude();
          this.updateDataUnits(null);
          this.updateOutput("Displaying: Masking input");
        } else {
          this.updateOutput("Masking input not available - click Prepare first");
        }
        return;
      }

      // Handle mask (local data, not from pipeline)
      if (stage === 'mask') {
        if (this.currentMaskData || this.maskController.currentMaskData) {
          if (!this.currentMaskData) {
            this.currentMaskData = this.maskController.currentMaskData;
          }
          await this.displayCurrentMask();
          this.updateDataUnits(null);
          this.updateOutput("Displaying: Brain Mask");
        } else {
          this.updateOutput("Mask not available - generate one first");
        }
        return;
      }

      // Check if we have cached results
      if (this.results[stage]?.file) {
        const description = this.results[stage].description || stage;
        this.updateOutput(`Displaying ${description}...`);
        await this.loadAndVisualizeFile(this.results[stage].file, description);
        // Re-apply saved display range
        const displayRange = this.results[stage].displayRange;
        if (displayRange && this.nv.volumes?.length > 0) {
          const vol = this.nv.volumes[0];
          vol.cal_min = displayRange[0];
          vol.cal_max = displayRange[1];
          this.nv.updateGLVolume();
        }
        return;
      }

      // No cached result available
      this.updateOutput(`${stage} not available - run the pipeline first`);

    } catch (error) {
      this.updateOutput(`Error showing ${stage}: ${error.message}`);
    }
  }

  // Display stage data as it arrives during pipeline processing
  async displayLiveStageData(data) {
    try {
      const { stage, data: stageBytes, description, displayRange } = data;

      // Show the stage buttons section as soon as first result arrives
      this.showStageButtons();

      // Add/enable the button for this stage (with description for display name)
      this.addStageButton(stage, description);

      // Hide echo navigation - pipeline results are single 3D volumes, not multi-echo
      this.hideEchoNavigation();

      // Create file from bytes
      const blob = new Blob([stageBytes], { type: 'application/octet-stream' });
      const file = new File([blob], `${stage}.nii`, { type: 'application/octet-stream' });

      // Load in viewer
      await this.loadAndVisualizeFile(file, description);

      // Apply custom display range if provided (e.g. robust percentile range for T2*/R2*)
      if (displayRange && this.nv.volumes?.length > 0) {
        const vol = this.nv.volumes[0];
        vol.cal_min = displayRange[0];
        vol.cal_max = displayRange[1];
        this.nv.updateGLVolume();
      }

      // Cache the result with description and display range
      this.results[stage] = { file: file, path: `${stage}.nii`, description: description, displayRange: displayRange };

      this.updateOutput(`Displaying: ${description}`);
    } catch (error) {
      this.updateOutput(`Error displaying live data: ${error.message}`);
    }
  }

  // Cache stage data without displaying (for auxiliary outputs like vasculature mask)
  cacheStageData(data) {
    try {
      const { stage, data: stageBytes, description } = data;

      // Show the stage buttons section
      this.showStageButtons();

      // Add/enable the button for this stage (with description for display name)
      this.addStageButton(stage, description);

      // Create file from bytes
      const blob = new Blob([stageBytes], { type: 'application/octet-stream' });
      const file = new File([blob], `${stage}.nii`, { type: 'application/octet-stream' });

      // Cache the result (but don't display)
      this.results[stage] = { file: file, path: `${stage}.nii`, description: description };

      // Silently cached — no console message needed
    } catch (error) {
      this.updateOutput(`Error caching data: ${error.message}`);
    }
  }

  /**
   * Download the currently displayed volume as a NIfTI file
   */
  downloadCurrentVolume() {
    if (!this.nv.volumes || this.nv.volumes.length === 0) {
      this.updateOutput("No volume loaded to download");
      return;
    }

    const vol = this.nv.volumes[0];
    const name = vol.name || 'volume';
    const baseName = name.replace(/\.(nii|nii\.gz)$/i, '');

    // Create NIfTI from volume data
    const niftiBuffer = this.createNiftiFromVolume(vol);

    // Download
    const blob = new Blob([niftiBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseName}.nii`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.updateOutput(`Downloaded: ${baseName}.nii`);
  }

  /**
   * Save a screenshot of the NiiVue viewer as PNG
   */
  saveScreenshot() {
    if (!this.nv) {
      this.updateOutput("Viewer not initialized");
      return;
    }

    // Generate filename based on current volume or timestamp
    let filename = 'niivue_screenshot.png';
    if (this.nv.volumes && this.nv.volumes.length > 0) {
      const vol = this.nv.volumes[0];
      const name = vol.name || 'volume';
      const baseName = name.replace(/\.(nii|nii\.gz)$/i, '');
      filename = `${baseName}_screenshot.png`;
    }

    this.nv.saveScene(filename);
    this.updateOutput(`Screenshot saved: ${filename}`);
  }

  /**
   * Create a NIfTI buffer from a NiiVue volume
   */
  createNiftiFromVolume(vol) {
    const hdr = vol.hdr;
    const img = vol.img;

    // Determine data type and bytes per voxel
    let datatype = 16;  // FLOAT32 by default
    let bitpix = 32;
    let bytesPerVoxel = 4;

    if (img instanceof Float64Array) {
      datatype = 64;  // FLOAT64
      bitpix = 64;
      bytesPerVoxel = 8;
    } else if (img instanceof Int16Array) {
      datatype = 4;   // INT16
      bitpix = 16;
      bytesPerVoxel = 2;
    } else if (img instanceof Uint8Array) {
      datatype = 2;   // UINT8
      bitpix = 8;
      bytesPerVoxel = 1;
    }

    const headerSize = 352;
    const dataSize = img.length * bytesPerVoxel;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);

    // sizeof_hdr
    view.setInt32(0, 348, true);

    // dim array
    const dims = hdr.dims || [3, vol.dims[1], vol.dims[2], vol.dims[3], 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setInt16(40 + i * 2, dims[i] || 0, true);
    }

    // datatype and bitpix
    view.setInt16(70, datatype, true);
    view.setInt16(72, bitpix, true);

    // pixdim
    const pixdim = hdr.pixDims || [1, 1, 1, 1, 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setFloat32(76 + i * 4, pixdim[i] || 1, true);
    }

    // vox_offset
    view.setFloat32(108, headerSize, true);

    // scl_slope and scl_inter
    view.setFloat32(112, hdr.scl_slope || 1, true);
    view.setFloat32(116, hdr.scl_inter || 0, true);

    // xyzt_units
    view.setUint8(123, 10);  // mm + sec

    // qform_code and sform_code
    view.setInt16(252, hdr.qform_code || 1, true);
    view.setInt16(254, hdr.sform_code || 1, true);

    // Affine matrix
    if (hdr.affine) {
      for (let i = 0; i < 4; i++) {
        view.setFloat32(280 + i * 4, hdr.affine[0][i] || 0, true);
        view.setFloat32(296 + i * 4, hdr.affine[1][i] || 0, true);
        view.setFloat32(312 + i * 4, hdr.affine[2][i] || 0, true);
      }
    }

    // magic
    view.setUint8(344, 0x6E);  // 'n'
    view.setUint8(345, 0x2B);  // '+'
    view.setUint8(346, 0x31);  // '1'
    view.setUint8(347, 0x00);

    // Copy image data
    const dataView = new Uint8Array(buffer, headerSize);
    const imgBytes = new Uint8Array(img.buffer, img.byteOffset, img.byteLength);
    dataView.set(imgBytes);

    return buffer;
  }

  updateOutput(message) {
    const consoleOutput = document.getElementById('consoleOutput');
    if (consoleOutput) {
      const time = new Date().toLocaleTimeString('en-US', { hour12: false });
      const line = document.createElement('div');
      line.className = 'console-line';
      line.innerHTML = `<span class="console-time">[${time}]</span> <span class="console-message">${message}</span>`;
      consoleOutput.appendChild(line);
      // Auto-scroll to bottom
      consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
    console.log(message);
  }

  /**
   * Auto-detect optimal threshold using Otsu's method and set the slider
   * Delegates computation to imported ThresholdUtils module
   */
  /**
   * Auto-detect optimal threshold using Otsu's method
   * Delegates to MaskController
   */
  autoDetectThreshold() {
    if (!this.preparedMagnitudeData) {
      this.updateOutput("Please click Prepare first");
      return;
    }

    this.updateOutput("Computing optimal threshold (Otsu)...");

    // Sync prepared data to controller
    this.maskController.preparedMagnitudeData = this.preparedMagnitudeData;
    this.maskController.preparedMagnitudeMax = this.preparedMagnitudeMax;

    const result = this.maskController.computeOtsuThreshold();

    if (result.error) {
      this.updateOutput(`Cannot compute threshold: ${result.error}`);
      return;
    }

    const clampedPercent = result.thresholdPercent;

    // Update slider and display
    const slider = document.getElementById('maskThreshold');
    if (slider) {
      slider.value = clampedPercent;
      this.maskThreshold = clampedPercent;
      this.maskController.setMaskThreshold(clampedPercent);
      document.getElementById('thresholdLabel').textContent = `Threshold (${clampedPercent}%)`;
    }

    this.updateOutput(`Otsu threshold: ${clampedPercent}% (${result.thresholdValue.toFixed(1)})`);

    // Only trigger mask preview if threshold slider is enabled (user has clicked Threshold button)
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider && !thresholdSlider.disabled && this.magnitudeData && !this.maskUpdating) {
      this.updateMaskPreview();
    }
  }

  /**
   * Run BET brain extraction
   * Delegates to MaskController
   */
  async runBET() {
    // Track BET as mask generator
    const fi = this.betSettings?.fractionalIntensity ?? 0.5;
    this.maskOpsHistory = [`bet:${fi}`];

    // Disable threshold slider since user chose BET-based masking
    this.setThresholdSliderEnabled(false);

    // Sync state to controller
    this.maskController.magnitudeFileBytes = this.magnitudeFileBytes;
    this.maskController.magnitudeVolume = this.magnitudeVolume;
    this.maskController.magnitudeData = this.magnitudeData;
    this.maskController.magnitudeMax = this.magnitudeMax;
    this.maskController.preparedMagnitudeData = this.preparedMagnitudeData;
    this.maskController.maskDims = this.maskDims;

    // Get magnitude files from unified buckets
    const magnitudeFilesForBET = this.fileIOController.buckets.magnitude;

    await this.maskController.runBET({
      magnitudeFiles: magnitudeFilesForBET,
      betSettings: this.betSettings,
      createNiftiHeaderFromVolume: (vol) => this.createNiftiHeaderFromVolume(vol),
      onComplete: async () => {
        // Sync state from controller
        this.currentMaskData = this.maskController.currentMaskData;
        this.originalMaskData = this.maskController.originalMaskData;
        this.maskDims = this.maskController.maskDims;
        this.magnitudeVolume = this.maskController.magnitudeVolume;
        this.magnitudeData = this.maskController.magnitudeData;
        this.magnitudeMax = this.maskController.magnitudeMax;
        this.magnitudeFileBytes = this.maskController.magnitudeFileBytes;

        // Apply post-BET erosions
        const erosions = this.betSettings.erosions || 0;
        if (erosions > 0) {
          this.updateOutput(`Applying ${erosions} erosion step(s)...`);
          for (let i = 0; i < erosions; i++) {
            this.erodeMask3D();
          }
          await this.displayCurrentMask();
          this.updateOutput(`BET mask complete with ${erosions} erosion(s)`);
        }

        // Show morphological operations panel
        const opsPanel = document.getElementById('maskOperations');
        if (opsPanel) opsPanel.style.display = 'block';

        // Add mask to Results section
        this.showStageButtons();
        this.addStageButton('mask', 'Brain Mask');

        // Update run button state (mask is now available)
        this.updateEchoInfo();
      },
      onError: () => {
        this.updateEchoInfo();
      }
    });
  }

  /**
   * Handle BET completion - delegates to MaskController
   */
  async handleBETComplete(data) {
    await this.maskController.handleBETComplete(data, () => {
      // Sync state from controller
      this.currentMaskData = this.maskController.currentMaskData;
      this.originalMaskData = this.maskController.originalMaskData;

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Add mask to Results section
      this.showStageButtons();
      this.addStageButton('mask', 'Brain Mask');

      // Update run button state
      this.updateEchoInfo();
    });
  }

  // Calculate dynamic defaults based on voxel size (matches QSM.jl)
  getVoxelBasedDefaults() {
    return window.QSMConfig.getVoxelBasedDefaults(this.voxelSize || [1, 1, 1], this.maskDims);
  }

  // Apply voxel-based defaults to pipeline settings for any null values.
  // Called when voxel size becomes available (file upload, mask preparation).
  applyVoxelDefaults() {
    if (!this.voxelSize) return;
    const defaults = this.getVoxelBasedDefaults();
    const s = this.pipelineSettings;
    if (s.vsharp.max_radius == null) s.vsharp.max_radius = defaults.vsharpMaxRadius;
    if (s.vsharp.min_radius == null) s.vsharp.min_radius = defaults.vsharpMinRadius;
    if (s.ismv.radius == null) s.ismv.radius = defaults.ismv_radius;
    if (s.pdf.maxit == null) s.pdf.maxit = defaults.pdfMaxit;
  }

  // Pipeline Settings Modal - delegates to PipelineSettingsController
  openPipelineSettingsModal() {
    if (!this.pipelineSettingsController) return;
    const defaults = this.getVoxelBasedDefaults();
    const nEchoes = this.fileIOController?.buckets?.phase?.filter(f => f.file)?.length || 0;
    const inputMode = this.fileIOController?.getInputMode() || 'raw';
    const hasMagnitude = this.fileIOController?.buckets?.magnitude?.length > 0
      || this.preparedMagnitudeData !== null;
    this.pipelineSettingsController.setInputMode(inputMode);
    this.pipelineSettingsController.open(this.pipelineSettings, defaults, nEchoes, hasMagnitude);
    this.updateEchoInfo();
  }

  closePipelineSettingsModal() {
    if (this.pipelineSettingsController) {
      this.pipelineSettingsController.close();
    }
  }

  resetPipelineSettings() {
    if (!this.pipelineSettingsController) return;
    const defaults = this.getVoxelBasedDefaults();
    this.pipelineSettingsController.reset(defaults);
  }

  savePipelineSettings() {
    if (!this.pipelineSettingsController) return;
    const nEchoes = this.fileIOController?.buckets?.phase?.filter(f => f.file)?.length || 0;
    this.pipelineSettings = this.pipelineSettingsController.save(nEchoes);
    this.closePipelineSettingsModal();
    // Sync sidebar dropdowns and update visibility
    this.syncSidebarFromSettings();
    this.updateInputParamsVisibility();
    this.updateEchoInfo();
  }

  runPipelineFromSidebar() {
    // Save current settings from modal if it's open
    const modal = document.getElementById('pipelineSettingsModal');
    if (modal && modal.classList.contains('active')) {
      this.savePipelineSettings();
    }

    // Collapse pipeline section, open results
    const pipelineSection = document.getElementById('pipelineSection');
    const resultsSection = document.getElementById('stage-buttons');
    if (pipelineSection) pipelineSection.classList.add('collapsed');
    if (resultsSection) {
      resultsSection.classList.remove('collapsed');
      resultsSection.classList.remove('hidden');
    }

    this.runRomeoQSM();
  }

  async runSWI() {
    const mode = this.fileIOController.getInputMode();
    if (mode !== 'raw') {
      this.updateOutput("SWI requires raw magnitude + phase data");
      return;
    }

    const magCount = this.fileIOController.buckets.magnitude.length;
    const phaseCount = this.fileIOController.buckets.phase.length;

    if (magCount === 0 || phaseCount === 0) {
      this.updateOutput("Please upload both magnitude and phase files");
      return;
    }

    try {
      const magnitudeBuffers = [];
      const phaseBuffers = [];

      // Only need first echo for SWI
      const magFile = this.fileIOController.buckets.magnitude[0]?.file;
      const phaseFile = this.fileIOController.buckets.phase[0]?.file;

      if (magFile && phaseFile) {
        magnitudeBuffers.push(await magFile.arrayBuffer());
        phaseBuffers.push(await phaseFile.arrayBuffer());
      }

      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        customMaskBuffer = this.createMaskNifti(this.currentMaskData);
      }

      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      await this.pipelineExecutor.initialize();
      this.pipelineExecutor.pipelineRunning = true;
      this.updateOutput("Starting SWI pipeline...");

      this.pipelineExecutor.getWorker().postMessage({
        type: 'runSWI',
        data: {
          magnitudeBuffers,
          phaseBuffers,
          maskThreshold: this.maskThreshold,
          customMaskBuffer,
          preparedMagnitude,
          pipelineSettings: this.pipelineSettings
        }
      });

      document.getElementById('cancelPipeline').disabled = false;
      document.getElementById('runSWI').disabled = true;

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      console.error(error);
    }
  }

  async runT2starR2star() {
    const mode = this.fileIOController.getInputMode();
    if (mode !== 'raw') {
      this.updateOutput("T2*/R2* requires raw magnitude data (current mode: " + mode + ")");
      return;
    }

    const magCount = this.fileIOController.buckets.magnitude.length;
    if (magCount < 3) {
      this.updateOutput(`T2*/R2* mapping requires 3+ echo magnitudes (found ${magCount})`);
      return;
    }

    try {
      const magnitudeBuffers = [];
      for (const entry of this.fileIOController.buckets.magnitude) {
        if (entry?.file) {
          magnitudeBuffers.push(await entry.file.arrayBuffer());
        }
      }

      const echoTimes = this.getEchoTimesFromInputs();
      if (echoTimes.length < 3) {
        this.updateOutput(`T2*/R2* mapping requires echo times for 3+ echoes (found ${echoTimes.length}). Check that JSON sidecars with EchoTime are loaded.`);
        return;
      }

      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        customMaskBuffer = this.createMaskNifti(this.currentMaskData);
      }

      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      await this.pipelineExecutor.initialize();
      this.pipelineExecutor.pipelineRunning = true;
      this.updateOutput("Starting T2*/R2* mapping...");

      this.pipelineExecutor.getWorker().postMessage({
        type: 'runT2starR2star',
        data: {
          magnitudeBuffers,
          maskThreshold: this.maskThreshold,
          customMaskBuffer,
          preparedMagnitude,
          echoTimes
        }
      });

      document.getElementById('cancelPipeline').disabled = false;
      document.getElementById('runT2starR2star').disabled = true;

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      console.error(error);
    }
  }

  // BET Settings Modal
  openBetSettingsModal() {
    const hasMag = this.fileIOController.buckets.magnitude.length > 0;

    if (!hasMag) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    // Populate form with current settings
    document.getElementById('betFractionalIntensity').value = this.betSettings.fractionalIntensity;
    document.getElementById('betFractionalIntensityValue').textContent = this.betSettings.fractionalIntensity;
    document.getElementById('betIterations').value = this.betSettings.iterations;
    document.getElementById('betSubdivisions').value = this.betSettings.subdivisions;
    document.getElementById('betErosions').value = this.betSettings.erosions ?? 2;

    this.betModal?.open();
  }

  resetBetSettings() {
    // Reset to defaults
    document.getElementById('betFractionalIntensity').value = 0.5;
    document.getElementById('betFractionalIntensityValue').textContent = '0.5';
    document.getElementById('betIterations').value = 1000;
    document.getElementById('betSubdivisions').value = 4;
    document.getElementById('betErosions').value = 2;
  }

  runBetWithSettings() {
    // Save settings from form
    this.betSettings = {
      fractionalIntensity: parseFloat(document.getElementById('betFractionalIntensity').value),
      iterations: parseInt(document.getElementById('betIterations').value),
      subdivisions: parseInt(document.getElementById('betSubdivisions').value),
      erosions: parseInt(document.getElementById('betErosions').value) || 0
    };

    this.betModal?.close();
    this.runBET();
  }

  // --- Command Preview ---

  showCommandPreview() {
    // Sync sidebar SWI settings to pipeline settings before generating command
    if (this.pipelineSettings.swi) {
      this.pipelineSettings.swi.scaling = document.getElementById('sidebarSwiScaling')?.value || 'tanh';
      this.pipelineSettings.swi.strength = parseFloat(document.getElementById('sidebarSwiStrength')?.value) || 4;
      this.pipelineSettings.swi.mip_window = parseInt(document.getElementById('sidebarSwiMipWindow')?.value) || 7;
      this.pipelineSettings.swi.hp_sigma = [
        parseFloat(document.getElementById('sidebarSwiHpSigmaX')?.value) || 4,
        parseFloat(document.getElementById('sidebarSwiHpSigmaY')?.value) || 4,
        parseFloat(document.getElementById('sidebarSwiHpSigmaZ')?.value) || 0,
      ];
    }
    const configJson = buildConfigJson(this.pipelineSettings, {
      doSwi: !!this.results?.swi?.file,
      doT2star: !!this.results?.t2star?.file,
      doR2star: !!this.results?.r2star?.file,
    });
    const maskSource = this.maskPrepSettings?.source || 'phase_quality';
    this._lastToml = null; // populated async by configTomlResult (for the Download .toml button)

    // Show modal immediately with loading state
    const cmdEl = document.getElementById('commandPreviewText');
    const methodsRendered = document.getElementById('methodsPreviewRendered');
    const methodsRaw = document.getElementById('methodsPreviewRaw');
    if (cmdEl) cmdEl.textContent = 'Generating...';
    if (methodsRendered) methodsRendered.innerHTML = '<em>Generating...</em>';
    this.switchExportTab('command');
    this.commandPreviewModal?.open();

    // Ask worker to generate command and methods via WASM
    const worker = this.pipelineExecutor?.worker;
    if (!worker) { if (cmdEl) cmdEl.textContent = 'ERROR: Worker not available'; return; }

    const maskSection = maskSectionString(this.maskOpsHistory, maskSource);
    const handler = (e) => {
      if (e.data.type === 'commandResult') {
        if (cmdEl) cmdEl.textContent = e.data.result;
      } else if (e.data.type === 'methodsResult') {
        const raw = e.data.result;
        if (methodsRaw) methodsRaw.textContent = raw;
        if (methodsRendered) methodsRendered.innerHTML = renderMarkdown(raw);
      } else if (e.data.type === 'configTomlResult') {
        this._lastToml = e.data.result; // last message back — safe to detach
        worker.removeEventListener('message', handler);
      }
    };
    worker.addEventListener('message', handler);
    worker.postMessage({ type: 'generateCommand', data: { configJson, maskSection } });
    worker.postMessage({ type: 'generateMethods', data: { configJson, maskSection } });
    worker.postMessage({ type: 'generateConfigToml', data: { configJson, maskSection } });
  }

  switchExportTab(tab) {
    const cmdPane = document.getElementById('exportCommandPane');
    const methodsPane = document.getElementById('exportMethodsPane');
    const cmdTab = document.getElementById('exportTabCommand');
    const methodsTab = document.getElementById('exportTabMethods');
    if (tab === 'methods') {
      cmdPane.style.display = 'none';
      methodsPane.style.display = '';
      cmdTab.classList.remove('active');
      methodsTab.classList.add('active');
    } else {
      cmdPane.style.display = '';
      methodsPane.style.display = 'none';
      cmdTab.classList.add('active');
      methodsTab.classList.remove('active');
    }
  }

  downloadSettingsToml() {
    if (!this._lastToml) { this.updateOutput('Settings TOML not ready yet — try again in a moment.'); return; }
    const bytes = new TextEncoder().encode(this._lastToml);
    this._downloadBuffer(bytes, 'qsmbly-settings.toml');
  }

  copyCommandToClipboard() {
    // Copy command text or raw markdown depending on active tab
    const cmdPane = document.getElementById('exportCommandPane');
    const el = cmdPane.style.display !== 'none'
      ? document.getElementById('commandPreviewText')
      : document.getElementById('methodsPreviewRaw');
    if (!el) return;
    navigator.clipboard.writeText(el.textContent).then(() => {
      const btn = document.getElementById('copyCommand');
      if (btn) {
        const original = btn.innerHTML;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.innerHTML = original; }, 1500);
      }
    });
  }
}

// Export the QSMApp class for ES module usage
export { QSMApp };

// Initialize the app - this will be called after NiiVue is loaded
function initQSMApp() {
  console.log('Initializing QSM App with NiiVue:', window.Niivue);
  window.app = new QSMApp();
}

// Auto-initialize when loaded as a module
// Wait for DOM and NiiVue to be ready
function waitForNiiVue(maxAttempts = 20, attempt = 0) {
  if (window.Niivue) {
    initQSMApp();
  } else if (attempt < maxAttempts) {
    setTimeout(() => waitForNiiVue(maxAttempts, attempt + 1), 100);
  } else {
    document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => waitForNiiVue());
} else {
  waitForNiiVue();
}