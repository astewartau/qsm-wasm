class QSMApp {
  constructor() {
    this.worker = null;
    this.workerReady = false;
    this.nv = new window.Niivue({
      loadingText: "",
      textHeight: 0.03,
      onLocationChange: (data) => {
        document.getElementById("intensity").innerHTML = data.string;
      }
    });
    this.currentFile = null;
    this.threshold = 75;
    this.progress = 0;

    // Smooth progress animation state
    this.targetProgress = 0;      // Actual progress from pipeline
    this.animatedProgress = 0;    // Displayed (animated) progress
    this.progressAnimationId = null;
    this.lastAnimationTime = 0;
    // Animation speed: move at ~0.67% per second (completes 100% in ~150s typical runtime)
    this.progressAnimationSpeed = 0.0067;

    // Multi-echo file storage
    this.multiEchoFiles = {
      magnitude: [],
      phase: [],
      json: [],
      echoTimes: [],
      combinedMagnitude: null,
      combinedPhase: null
    };

    // Processing results
    this.results = {
      magnitude: { path: null, file: null },
      phase: { path: null, file: null },
      mask: { path: null, file: null },
      B0: { path: null, file: null },
      bgRemoved: { path: null, file: null },
      final: { path: null, file: null }
    };

    // Pending stage requests
    this.pendingStageResolve = null;

    // Mask threshold (percentage of max magnitude)
    this.maskThreshold = 15;
    this.magnitudeData = null;  // Cached magnitude array for mask preview
    this.magnitudeMax = 0;

    // Mask editing state
    this.currentMaskData = null;      // Current edited mask (Float32Array)
    this.originalMaskData = null;     // Original threshold-based mask for reset
    this.maskDims = null;             // [nx, ny, nz] dimensions

    // Drawing state
    this.drawingEnabled = false;
    this.brushMode = 'add';           // 'add' or 'remove'
    this.brushSize = 2;
    this.savedCrosshairWidth = 1;     // Store crosshair width when hiding

    // Pipeline settings
    this.pipelineSettings = {
      unwrapMethod: 'romeo',  // 'romeo' or 'laplacian'
      unwrapMode: 'individual',  // 'individual' or 'temporal'
      romeo: { weighting: 'phase_snr' },
      backgroundRemoval: 'vsharp',  // 'vsharp' or 'smv'
      vsharp: { maxRadius: 18, minRadius: 2, threshold: 0.05 },
      smv: { radius: 5 },
      dipoleInversion: 'tv',  // 'tkd', 'tikhonov', 'tv', 'rts'
      tkd: { threshold: 0.15 },
      tikhonov: { lambda: 0.01, reg: 'identity' },
      tv: { lambda: 0.001, maxIter: 250, tol: 0.001 },
      rts: { delta: 0.15, mu: 100000, rho: 10, maxIter: 20 }
    };

    // BET settings
    this.betSettings = {
      fractionalIntensity: 0.5,
      iterations: 1000,
      subdivisions: 4
    };

    // Track last run settings for intelligent caching
    this.lastRunSettings = null;
    this.pipelineHasRun = false;

    this.init();
  }

  async init() {
    await this.setupViewer();
    this.setupUIControls();
    this.setupEventListeners();
    this.updateDownloadButtons();

    // Initialize file lists
    this.updateFileList('magnitude', []);
    this.updateFileList('phase', []);
    this.updateFileList('json', []);

    this.updateOutput("Ready. Upload magnitude, phase, and JSON files for each echo.");
  }

  setupWorker() {
    if (this.worker) return;

    this.worker = new Worker('js/qsm-worker.js');

    this.worker.onmessage = (e) => {
      const { type, ...data } = e.data;

      switch (type) {
        case 'progress':
          this.setProgress(data.value, data.text);
          break;

        case 'log':
          this.updateOutput(data.message);
          break;

        case 'error':
          this.updateOutput(`Error: ${data.message}`);
          this.setProgress(0, 'Failed');
          break;

        case 'initialized':
          this.workerReady = true;
          this.updateOutput("Pyodide initialized in worker");
          break;

        case 'complete':
          this.updateOutput("Pipeline completed successfully!");
          this.showStageButtons();
          this.showStage('final'); // Auto-display QSM result
          // Save settings for intelligent caching on next run
          this.lastRunSettings = JSON.parse(JSON.stringify(this.pipelineSettings));
          this.pipelineHasRun = true;
          break;

        case 'stageData':
          if (this.pendingStageResolve) {
            this.pendingStageResolve(data);
            this.pendingStageResolve = null;
          }
          break;
      }
    };

    this.worker.onerror = (e) => {
      this.updateOutput(`Worker error: ${e.message}`);
      console.error('Worker error:', e);
    };
  }

  async initializeWorker() {
    this.setupWorker();

    if (this.workerReady) return;

    this.updateOutput("Initializing Pyodide in background...");

    // Fetch ROMEO code
    const romeoCode = await fetch('./python/romeo_python.py').then(r => r.text());

    // Send init message to worker
    this.worker.postMessage({
      type: 'init',
      data: { romeoCode }
    });

    // Wait for initialization
    return new Promise((resolve) => {
      const checkReady = setInterval(() => {
        if (this.workerReady) {
          clearInterval(checkReady);
          resolve();
        }
      }, 100);
    });
  }

  async setupViewer() {
    await this.nv.attachTo("gl1");
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
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

    // Start animation if not already running
    if (!this.progressAnimationId && value > 0 && value < 1) {
      this.lastAnimationTime = performance.now();
      this.animateProgress();
    }

    // If complete, jump to 100% and stop animation
    if (value >= 1) {
      this.animatedProgress = 1;
      this.updateProgressBar();
      this.stopProgressAnimation();
    }

    // If reset to 0, reset animation
    if (value === 0) {
      this.animatedProgress = 0;
      this.updateProgressBar();
      this.stopProgressAnimation();
    }
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
    const fill = document.getElementById('progressFill');
    if (fill) {
      fill.style.width = `${this.animatedProgress * 100}%`;
    }
  }

  stopProgressAnimation() {
    if (this.progressAnimationId) {
      cancelAnimationFrame(this.progressAnimationId);
      this.progressAnimationId = null;
    }
  }

  setupEventListeners() {
    // Multi-echo file inputs
    document.getElementById('magnitudeFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'magnitude');
    });
    
    document.getElementById('phaseFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'phase');
    });
    
    document.getElementById('jsonFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'json');
    });

    // Echo time Tagify input
    this.setupEchoTagify();

    // Processing buttons
    document.getElementById('run').addEventListener('click', () => this.openPipelineSettingsModal());
    document.getElementById('vis_magnitude').addEventListener('click', () => this.visualizeMagnitude());
    document.getElementById('vis_phase').addEventListener('click', () => this.visualizePhase());

    // Stage navigation
    document.getElementById('showMagnitude').addEventListener('click', () => this.showStage('magnitude'));
    document.getElementById('showPhase').addEventListener('click', () => this.showStage('phase'));
    document.getElementById('showMask').addEventListener('click', () => this.showStage('mask'));
    document.getElementById('showB0').addEventListener('click', () => this.showStage('B0'));
    document.getElementById('showBgRemoved').addEventListener('click', () => this.showStage('bgRemoved'));
    document.getElementById('showDipoleInversed').addEventListener('click', () => this.showStage('final'));

    // Download buttons
    document.getElementById('downloadMagnitude').addEventListener('click', () => this.downloadStage('magnitude'));
    document.getElementById('downloadPhase').addEventListener('click', () => this.downloadStage('phase'));
    document.getElementById('downloadMask').addEventListener('click', () => this.downloadStage('mask'));
    document.getElementById('downloadB0').addEventListener('click', () => this.downloadStage('B0'));
    document.getElementById('downloadBgRemoved').addEventListener('click', () => this.downloadStage('bgRemoved'));
    document.getElementById('downloadDipoleInversed').addEventListener('click', () => this.downloadStage('final'));

    // Mask threshold slider with debounce
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        this.maskThreshold = parseInt(e.target.value);
        document.getElementById('thresholdValue').textContent = `${this.maskThreshold}%`;

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

    // Preview mask button
    const previewMaskBtn = document.getElementById('previewMask');
    if (previewMaskBtn) {
      previewMaskBtn.addEventListener('click', () => this.previewMask());
    }

    // BET brain extraction button - opens settings modal
    document.getElementById('runBET')?.addEventListener('click', () => this.openBetSettingsModal());

    // Pipeline settings modal
    document.getElementById('closePipelineSettings')?.addEventListener('click', () => this.closePipelineSettingsModal());
    document.getElementById('resetPipelineSettings')?.addEventListener('click', () => this.resetPipelineSettings());
    document.getElementById('runPipelineWithSettings')?.addEventListener('click', () => this.runPipelineWithSettings());

    // BET settings modal
    document.getElementById('closeBetSettings')?.addEventListener('click', () => this.closeBetSettingsModal());
    document.getElementById('resetBetSettings')?.addEventListener('click', () => this.resetBetSettings());
    document.getElementById('runBetWithSettings')?.addEventListener('click', () => this.runBetWithSettings());

    // BET fractional intensity slider value display
    document.getElementById('betFractionalIntensity')?.addEventListener('input', (e) => {
      document.getElementById('betFractionalIntensityValue').textContent = e.target.value;
    });

    // Close modals on overlay click
    document.getElementById('pipelineSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'pipelineSettingsModal') this.closePipelineSettingsModal();
    });
    document.getElementById('betSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'betSettingsModal') this.closeBetSettingsModal();
    });

    // Method selection dropdowns - show/hide appropriate settings
    document.getElementById('unwrapMethod')?.addEventListener('change', (e) => {
      const isRomeo = e.target.value === 'romeo';
      document.getElementById('romeoSettings').style.display = isRomeo ? 'block' : 'none';
      document.getElementById('laplacianSettings').style.display = isRomeo ? 'none' : 'block';
    });

    document.getElementById('bgRemovalMethod')?.addEventListener('change', (e) => {
      const isVsharp = e.target.value === 'vsharp';
      document.getElementById('vsharpSettings').style.display = isVsharp ? 'block' : 'none';
      document.getElementById('smvSettings').style.display = isVsharp ? 'none' : 'block';
    });

    document.getElementById('dipoleMethod')?.addEventListener('change', (e) => {
      const method = e.target.value;
      document.getElementById('tkdSettings').style.display = method === 'tkd' ? 'block' : 'none';
      document.getElementById('tikhonovSettings').style.display = method === 'tikhonov' ? 'block' : 'none';
      document.getElementById('tvSettings').style.display = method === 'tv' ? 'block' : 'none';
      document.getElementById('rtsSettings').style.display = method === 'rts' ? 'block' : 'none';
    });

    // Morphological operation buttons
    document.getElementById('fillHoles')?.addEventListener('click', async () => {
      this.updateOutput("Filling holes in mask...");
      this.fillHoles3D();
      await this.displayCurrentMask();
      this.updateOutput("Holes filled");
    });

    document.getElementById('erodeMask')?.addEventListener('click', async () => {
      this.updateOutput("Eroding mask...");
      this.erodeMask3D();
      await this.displayCurrentMask();
      this.updateOutput("Mask eroded");
    });

    document.getElementById('dilateMask')?.addEventListener('click', async () => {
      this.updateOutput("Dilating mask...");
      this.dilateMask3D();
      await this.displayCurrentMask();
      this.updateOutput("Mask dilated");
    });

    document.getElementById('resetMask')?.addEventListener('click', async () => {
      this.updateOutput("Clearing mask...");
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
      this.brushSize = parseInt(e.target.value);
      document.getElementById('brushSizeValue').textContent = this.brushSize;
      if (this.drawingEnabled) {
        this.nv.setPenValue(this.brushMode === 'add' ? 1 : 0, false);
        this.nv.opts.penSize = this.brushSize;
      }
    });

    document.getElementById('undoDraw')?.addEventListener('click', () => {
      this.nv.drawUndo();
    });

    document.getElementById('applyDrawing')?.addEventListener('click', async () => {
      await this.applyDrawingToMask();
    });
  }

  async handleMultipleFiles(event, type) {
    const files = Array.from(event.target.files);

    // Store files
    this.multiEchoFiles[type] = files.map(file => ({
      file: file,
      name: file.name
    }));

    // Update UI
    this.updateFileList(type, this.multiEchoFiles[type]);

    // Process JSON files immediately to extract echo times
    if (type === 'json') {
      await this.processJsonFiles(files);
    }

    // Enable mask section when magnitude files are loaded (but don't auto-generate mask)
    if (type === 'magnitude' && files.length > 0) {
      const maskSection = document.getElementById('maskSection');
      if (maskSection) {
        maskSection.classList.remove('section-disabled');
      }
      // Enable mask buttons
      document.getElementById('previewMask')?.removeAttribute('disabled');
      document.getElementById('runBET')?.removeAttribute('disabled');
      // Load magnitude into viewer for visualization
      this.visualizeMagnitude();
    }

    // Update echo information
    this.updateEchoInfo();
  }

  updateFileList(type, fileList) {
    const listElement = document.getElementById(`${type}List`);
    const fileDrop = listElement?.closest('.upload-group')?.querySelector('.file-drop');

    if (!listElement) {
      console.error(`File list element not found: ${type}List`);
      return;
    }

    listElement.innerHTML = '';

    if (fileList.length > 0) {
      fileDrop?.classList.add('has-files');
      fileList.forEach((fileData, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
          <span>${fileData.name}</span>
          <button class="file-remove" onclick="app.removeFile('${type}', ${index})">×</button>
        `;
        listElement.appendChild(fileItem);
      });

      // Update drop label
      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) {
        label.textContent = `${fileList.length} file${fileList.length > 1 ? 's' : ''} selected`;
      }
    } else {
      fileDrop?.classList.remove('has-files');
      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) {
        const defaults = {
          'magnitude': 'Drop files or click',
          'phase': 'Drop files or click',
          'json': 'Echo times (BIDS)'
        };
        label.textContent = defaults[type] || 'Drop files or click';
      }
    }
  }

  removeFile(type, index) {
    this.multiEchoFiles[type].splice(index, 1);
    this.updateFileList(type, this.multiEchoFiles[type]);
    this.updateEchoInfo();

    // Re-disable mask section if all magnitude files are removed
    if (type === 'magnitude' && this.multiEchoFiles.magnitude.length === 0) {
      const maskSection = document.getElementById('maskSection');
      if (maskSection) {
        maskSection.classList.add('section-disabled');
      }
      document.getElementById('previewMask')?.setAttribute('disabled', '');
      document.getElementById('runBET')?.setAttribute('disabled', '');
    }
  }

  async processJsonFiles(files) {
    console.log('Processing JSON files:', files);
    const echoTimes = [];
    
    for (const file of files) {
      try {
        const text = await file.text();
        const json = JSON.parse(text);
        
        console.log(`JSON file ${file.name} contents:`, json);
        console.log(`EchoTime field:`, json.EchoTime);
        console.log(`Available fields:`, Object.keys(json));
        
        // Extract echo time (in seconds, convert to ms)
        let echoTime = null;
        if (json.EchoTime) {
          echoTime = json.EchoTime * 1000; // Convert to ms
          console.log(`Found EchoTime: ${json.EchoTime}s -> ${echoTime}ms`);
        } else if (json.echo_time) {
          echoTime = json.echo_time * 1000;
          console.log(`Found echo_time: ${json.echo_time}s -> ${echoTime}ms`);
        } else if (json.TE) {
          echoTime = json.TE;
          console.log(`Found TE: ${json.TE}ms`);
        } else {
          console.warn(`No echo time found in ${file.name}. Available fields:`, Object.keys(json));
        }
        
        if (echoTime !== null) {
          echoTimes.push({
            file: file.name,
            echoTime: echoTime,
            json: json
          });
          console.log(`Added echo time for ${file.name}: ${echoTime}ms`);
        } else {
          console.error(`Could not extract echo time from ${file.name}`);
        }
      } catch (error) {
        console.error(`Error parsing JSON file ${file.name}:`, error);
      }
    }

    // Sort by echo time
    echoTimes.sort((a, b) => a.echoTime - b.echoTime);
    this.multiEchoFiles.echoTimes = echoTimes;

    console.log(`Final processed echo times:`, echoTimes);
    console.log(`Stored in multiEchoFiles:`, this.multiEchoFiles.echoTimes);

    // Populate the editable inputs
    this.populateEchoTimeInputs(echoTimes.map(et => et.echoTime));
  }

  updateEchoInfo() {
    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;
    const echoTimes = this.getEchoTimesFromInputs();
    const echoTimeCount = echoTimes.length;

    const echoBadge = document.getElementById('echoBadge');
    const runButton = document.getElementById('run');

    const isValid = magCount === phaseCount && magCount > 0;
    const hasEchoTimes = echoTimeCount > 0;
    const hasMask = this.currentMaskData !== null;
    const canRun = isValid && hasEchoTimes && hasMask;

    // Update echo badge
    if (echoBadge) {
      if (isValid && hasEchoTimes) {
        if (hasMask) {
          echoBadge.textContent = `${magCount} echo${magCount > 1 ? 'es' : ''} ready`;
          echoBadge.classList.add('ready');
        } else {
          echoBadge.textContent = 'Need mask';
          echoBadge.classList.remove('ready');
        }
      } else if (magCount > 0 || phaseCount > 0) {
        const issues = [];
        if (magCount !== phaseCount) issues.push('file mismatch');
        if (!hasEchoTimes) issues.push('no echo times');
        echoBadge.textContent = issues.join(', ') || 'Incomplete';
        echoBadge.classList.remove('ready');
      } else {
        echoBadge.textContent = 'No files loaded';
        echoBadge.classList.remove('ready');
      }
    }

    // Update echo times inputs (only if JSON was just loaded, not on every update)
    // The inputs are managed separately via populateEchoTimeInputs()

    // Update run button
    if (runButton) {
      runButton.disabled = !canRun;
      const btnText = runButton.querySelector('span');
      if (btnText) {
        if (canRun) {
          btnText.textContent = `Run Pipeline (${magCount} echoes)`;
        } else if (isValid && hasEchoTimes && !hasMask) {
          btnText.textContent = 'Run Pipeline (need mask)';
        } else {
          btnText.textContent = 'Run Pipeline';
        }
      }
    }
  }

  // Echo time Tagify management
  setupEchoTagify() {
    const input = document.getElementById('echoTimesTagify');
    if (!input || this.echoTagify) return;

    this.echoTagify = new Tagify(input, {
      delimiters: ',| ',
      pattern: /^[\d.]+$/,
      transformTag: (tagData) => {
        const num = parseFloat(tagData.value);
        if (!isNaN(num) && num > 0) {
          tagData.value = num.toFixed(2);
        }
      },
      validate: (tagData) => {
        const num = parseFloat(tagData.value);
        return !isNaN(num) && num > 0;
      },
      editTags: 1,
      placeholder: 'Type values...'
    });

    this.echoTagify.on('change', () => this.updateEchoInfo());
  }

  populateEchoTimeInputs(echoTimes) {
    if (!this.echoTagify) return;

    const tags = echoTimes.map(t => ({ value: t.toFixed(2) }));
    this.echoTagify.removeAllTags();
    this.echoTagify.addTags(tags);
    this.updateEchoInfo();
  }

  getEchoTimesFromInputs() {
    if (!this.echoTagify) return [];

    return this.echoTagify.value
      .map(tag => parseFloat(tag.value))
      .filter(n => !isNaN(n) && n > 0)
      .sort((a, b) => a - b);
  }

  async visualizeMagnitude() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded");
      return;
    }

    const file = this.multiEchoFiles.magnitude[0].file;
    await this.loadAndVisualizeFile(file, "Magnitude (Echo 1)");
  }

  async visualizePhase() {
    if (this.multiEchoFiles.phase.length === 0) {
      this.updateOutput("No phase files uploaded");
      return;
    }

    const file = this.multiEchoFiles.phase[0].file;
    await this.loadAndVisualizeFile(file, "Phase (Echo 1)");
  }

  async loadAndVisualizeFile(file, description) {
    try {
      this.updateOutput(`Loading ${description}...`);

      // Create a blob URL for NiiVue to load
      const url = URL.createObjectURL(file);

      // Load using loadVolumes which handles the URL properly
      await this.nv.loadVolumes([{ url: url, name: file.name }]);

      // Clean up the blob URL
      URL.revokeObjectURL(url);

      this.updateOutput(`${description} loaded`);
      this.currentFile = file;
    } catch (error) {
      this.updateOutput(`Error loading ${description}: ${error.message}`);
      console.error(error);
    }
  }

  async previewMask() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded");
      return;
    }

    try {
      this.updateOutput("Loading magnitude for mask preview...");
      const file = this.multiEchoFiles.magnitude[0].file;

      // Store the original file bytes for header copying
      this.magnitudeFileBytes = await file.arrayBuffer();

      // Load magnitude into NiiVue
      const url = URL.createObjectURL(file);
      await this.nv.loadVolumes([{ url: url, name: file.name }]);
      URL.revokeObjectURL(url);

      // Cache magnitude data for threshold updates
      if (this.nv.volumes.length > 0) {
        const vol = this.nv.volumes[0];
        this.magnitudeData = vol.img;
        // Find max without spread operator (avoids "too many arguments" for large arrays)
        let max = -Infinity;
        for (let i = 0; i < this.magnitudeData.length; i++) {
          if (this.magnitudeData[i] > max) max = this.magnitudeData[i];
        }
        this.magnitudeMax = max;
        // Store the full volume reference for header info
        this.magnitudeVolume = vol;

        // Update mask preview
        await this.updateMaskPreview();
      }

      this.updateOutput("Adjust threshold slider to refine mask");
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      console.error(error);
    }
  }

  async updateMaskPreview() {
    if (!this.magnitudeData || !this.nv.volumes.length || !this.magnitudeFileBytes) return;
    if (this.maskUpdating) return;  // Prevent concurrent updates

    this.maskUpdating = true;

    try {
      const threshold = (this.maskThreshold / 100) * this.magnitudeMax;
      const totalVoxels = this.magnitudeData.length;

      // Extract dimensions from NIfTI header
      const srcView = new DataView(this.magnitudeFileBytes);
      const nx = srcView.getInt16(42, true);  // dim[1]
      const ny = srcView.getInt16(44, true);  // dim[2]
      const nz = srcView.getInt16(46, true);  // dim[3]
      this.maskDims = [nx, ny, nz];

      // Create mask data from threshold
      const maskData = new Float32Array(totalVoxels);
      for (let i = 0; i < totalVoxels; i++) {
        maskData[i] = this.magnitudeData[i] > threshold ? 1 : 0;
      }

      // Store as both current and original (for reset)
      this.currentMaskData = maskData;
      this.originalMaskData = new Float32Array(maskData);

      // Display the mask
      await this.displayCurrentMask();

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Update run button state (mask is now available)
      this.updateEchoInfo();

    } catch (error) {
      console.error('Error updating mask preview:', error);
    } finally {
      this.maskUpdating = false;
    }
  }

  async displayCurrentMask() {
    if (!this.currentMaskData) return;

    // Count voxels in mask
    let maskCount = 0;
    const totalVoxels = this.currentMaskData.length;
    for (let i = 0; i < totalVoxels; i++) {
      if (this.currentMaskData[i] > 0) maskCount++;
    }

    // Update coverage display
    const coverage = ((maskCount / totalVoxels) * 100).toFixed(1);
    const coverageEl = document.getElementById('maskCoverage');
    if (coverageEl) {
      coverageEl.textContent = `Coverage: ${maskCount.toLocaleString()} / ${totalVoxels.toLocaleString()} voxels (${coverage}%)`;
    }

    // Close any existing drawing layer
    if (this.nv.drawBitmap) {
      this.nv.closeDrawing();
    }

    // Remove ALL existing overlays (everything except base volume)
    while (this.nv.volumes.length > 1) {
      await this.nv.removeVolumeByIndex(1);
    }

    // Create mask NIfTI by copying header from original file
    const maskNifti = this.createMaskNifti(this.currentMaskData);
    const maskBlob = new Blob([maskNifti], { type: 'application/octet-stream' });
    const maskUrl = URL.createObjectURL(maskBlob);

    await this.nv.addVolumeFromUrl({
      url: maskUrl,
      name: 'mask_preview.nii',
      colormap: 'red',
      opacity: 0.5
    });

    URL.revokeObjectURL(maskUrl);
  }

  // 3D morphological erosion (6-connected)
  erodeMask3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const src = this.currentMaskData;
    const dst = new Float32Array(src.length);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = x + y * nx + z * nx * ny;

          // Check if all 6 neighbors are inside mask
          if (src[idx] > 0) {
            let allNeighbors = true;

            // Check 6-connected neighbors
            if (x > 0 && src[idx - 1] === 0) allNeighbors = false;
            if (x < nx - 1 && src[idx + 1] === 0) allNeighbors = false;
            if (y > 0 && src[idx - nx] === 0) allNeighbors = false;
            if (y < ny - 1 && src[idx + nx] === 0) allNeighbors = false;
            if (z > 0 && src[idx - nx * ny] === 0) allNeighbors = false;
            if (z < nz - 1 && src[idx + nx * ny] === 0) allNeighbors = false;

            dst[idx] = allNeighbors ? 1 : 0;
          }
        }
      }
    }

    this.currentMaskData = dst;
  }

  // 3D morphological dilation (6-connected)
  dilateMask3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const src = this.currentMaskData;
    const dst = new Float32Array(src.length);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = x + y * nx + z * nx * ny;

          // Check if any of 6 neighbors is in mask
          if (src[idx] > 0) {
            dst[idx] = 1;
          } else {
            let anyNeighbor = false;

            if (x > 0 && src[idx - 1] > 0) anyNeighbor = true;
            if (x < nx - 1 && src[idx + 1] > 0) anyNeighbor = true;
            if (y > 0 && src[idx - nx] > 0) anyNeighbor = true;
            if (y < ny - 1 && src[idx + nx] > 0) anyNeighbor = true;
            if (z > 0 && src[idx - nx * ny] > 0) anyNeighbor = true;
            if (z < nz - 1 && src[idx + nx * ny] > 0) anyNeighbor = true;

            dst[idx] = anyNeighbor ? 1 : 0;
          }
        }
      }
    }

    this.currentMaskData = dst;
  }

  // Fill holes in 3D mask using flood fill from edges
  fillHoles3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const mask = this.currentMaskData;

    // Create a "visited from outside" array
    const outside = new Uint8Array(mask.length);

    // Use a queue for flood fill
    const queue = [];

    // Seed from all boundary voxels that are outside the mask
    // X boundaries
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const idx0 = 0 + y * nx + z * nx * ny;
        const idx1 = (nx - 1) + y * nx + z * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }
    // Y boundaries
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const idx0 = x + 0 * nx + z * nx * ny;
        const idx1 = x + (ny - 1) * nx + z * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }
    // Z boundaries
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx0 = x + y * nx + 0 * nx * ny;
        const idx1 = x + y * nx + (nz - 1) * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }

    // Flood fill from boundary
    const nxy = nx * ny;
    while (queue.length > 0) {
      const idx = queue.shift();
      const x = idx % nx;
      const y = Math.floor((idx % nxy) / nx);
      const z = Math.floor(idx / nxy);

      // Check 6-connected neighbors
      const neighbors = [];
      if (x > 0) neighbors.push(idx - 1);
      if (x < nx - 1) neighbors.push(idx + 1);
      if (y > 0) neighbors.push(idx - nx);
      if (y < ny - 1) neighbors.push(idx + nx);
      if (z > 0) neighbors.push(idx - nxy);
      if (z < nz - 1) neighbors.push(idx + nxy);

      for (const nidx of neighbors) {
        if (mask[nidx] === 0 && !outside[nidx]) {
          outside[nidx] = 1;
          queue.push(nidx);
        }
      }
    }

    // Fill holes: set all non-outside, non-mask voxels to 1
    const result = new Float32Array(mask.length);
    for (let i = 0; i < mask.length; i++) {
      result[i] = (mask[i] > 0 || !outside[i]) ? 1 : 0;
    }

    this.currentMaskData = result;
  }

  // Clear mask completely
  async clearMask() {
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Close any drawing layer
    if (this.nv.drawBitmap) {
      this.nv.closeDrawing();
    }

    // Remove mask overlay (keep only base volume)
    while (this.nv.volumes.length > 1) {
      await this.nv.removeVolumeByIndex(1);
    }

    // Update coverage display
    const coverageEl = document.getElementById('maskCoverage');
    if (coverageEl) {
      coverageEl.textContent = 'No mask';
    }

    // Update run button state (mask no longer available)
    this.updateEchoInfo();
  }

  // Toggle drawing mode on/off
  async toggleDrawingMode() {
    this.drawingEnabled = !this.drawingEnabled;

    const enableBtn = document.getElementById('enableDrawing');
    const addBtn = document.getElementById('brushAdd');
    const removeBtn = document.getElementById('brushRemove');
    const sizeControl = document.getElementById('brushSizeControl');
    const actionsDiv = document.getElementById('drawingActions');

    if (this.drawingEnabled) {
      // Need mask data and base volume
      if (!this.currentMaskData || this.nv.volumes.length === 0) {
        this.updateOutput("Please preview mask first before drawing");
        this.drawingEnabled = false;
        return;
      }

      // Enable drawing UI
      enableBtn.classList.add('active');
      addBtn.disabled = false;
      removeBtn.disabled = false;
      sizeControl.style.display = 'block';
      actionsDiv.style.display = 'grid';

      // Save current crosshair width and hide it
      this.savedCrosshairWidth = this.nv.opts.crosshairWidth;
      this.nv.opts.crosshairWidth = 0;
      this.nv.drawScene();  // Redraw to hide crosshair

      // Remove mask overlay - we'll use drawing layer instead
      while (this.nv.volumes.length > 1) {
        await this.nv.removeVolumeByIndex(1);
      }

      // Create drawing and load current mask into it
      this.nv.createEmptyDrawing();

      // Copy current mask to drawing bitmap
      const totalVoxels = this.currentMaskData.length;
      for (let i = 0; i < totalVoxels; i++) {
        this.nv.drawBitmap[i] = this.currentMaskData[i] > 0 ? 1 : 0;
      }

      // Refresh the drawing display
      this.nv.refreshDrawing(true);

      // Enable drawing mode
      this.nv.setDrawingEnabled(true);
      this.nv.opts.penSize = this.brushSize;
      this.nv.setDrawOpacity(0.5);

      // Set pen value for add mode (1)
      this.nv.setPenValue(1, false);
      this.brushMode = 'add';

      // Update UI to show add mode active
      addBtn.classList.add('active');
      removeBtn.classList.remove('active');

      this.updateOutput("Draw mode: DRAG to add, switch to Remove & drag to erase. Click Apply when done.");
    } else {
      // Disable drawing
      enableBtn.classList.remove('active');
      addBtn.disabled = true;
      removeBtn.disabled = true;
      addBtn.classList.remove('active');
      removeBtn.classList.remove('active');
      sizeControl.style.display = 'none';
      actionsDiv.style.display = 'none';

      // Restore crosshair
      if (this.savedCrosshairWidth !== undefined) {
        this.nv.opts.crosshairWidth = this.savedCrosshairWidth;
      }

      this.nv.setDrawingEnabled(false);
      this.nv.closeDrawing();

      // Redisplay the mask as overlay
      await this.displayCurrentMask();

      this.updateOutput("Drawing mode disabled");
    }
  }

  // Set brush mode (add or remove)
  setBrushMode(mode) {
    this.brushMode = mode;

    const addBtn = document.getElementById('brushAdd');
    const removeBtn = document.getElementById('brushRemove');

    addBtn.classList.toggle('active', mode === 'add');
    removeBtn.classList.toggle('active', mode === 'remove');

    // Pen value: 1 for adding to mask, 0 for erasing from mask
    // NiiVue uses 0 as the erase value
    const penValue = mode === 'add' ? 1 : 0;
    this.nv.setPenValue(penValue, false);

    this.updateOutput(`Brush: ${mode === 'add' ? 'Add (paint)' : 'Remove (erase)'}`);
  }

  // Apply the drawing to the current mask
  async applyDrawingToMask() {
    if (!this.currentMaskData || !this.maskDims) {
      this.updateOutput("No mask data to apply drawing to");
      return;
    }

    try {
      const drawBitmap = this.nv.drawBitmap;

      if (!drawBitmap || drawBitmap.length === 0) {
        this.updateOutput("No drawing to apply");
        return;
      }

      // Copy drawing bitmap directly to mask
      // The drawing IS the mask now, so just copy it back
      const totalVoxels = this.currentMaskData.length;
      let maskCount = 0;

      for (let i = 0; i < Math.min(drawBitmap.length, totalVoxels); i++) {
        this.currentMaskData[i] = drawBitmap[i] > 0 ? 1 : 0;
        if (drawBitmap[i] > 0) maskCount++;
      }

      // Exit drawing mode and show the mask as overlay
      this.drawingEnabled = false;

      // Update UI
      document.getElementById('enableDrawing')?.classList.remove('active');
      document.getElementById('brushAdd').disabled = true;
      document.getElementById('brushRemove').disabled = true;
      document.getElementById('brushAdd')?.classList.remove('active');
      document.getElementById('brushRemove')?.classList.remove('active');
      document.getElementById('brushSizeControl').style.display = 'none';
      document.getElementById('drawingActions').style.display = 'none';

      // Restore crosshair
      if (this.savedCrosshairWidth !== undefined) {
        this.nv.opts.crosshairWidth = this.savedCrosshairWidth;
      }

      // Close drawing and display mask as overlay
      this.nv.closeDrawing();
      this.nv.setDrawingEnabled(false);
      await this.displayCurrentMask();

      const coverage = ((maskCount / totalVoxels) * 100).toFixed(1);
      this.updateOutput(`Mask updated: ${maskCount.toLocaleString()} voxels (${coverage}%)`);

      // Update run button state (mask may have been created/modified)
      this.updateEchoInfo();
    } catch (error) {
      this.updateOutput(`Error applying drawing: ${error.message}`);
      console.error(error);
    }
  }

  createMaskNifti(maskData) {
    // Copy header from original magnitude file and replace data
    const srcBytes = new Uint8Array(this.magnitudeFileBytes);
    const srcView = new DataView(this.magnitudeFileBytes);

    // Get vox_offset from original header (offset 108, float32)
    const voxOffset = srcView.getFloat32(108, true);
    const headerSize = Math.ceil(voxOffset);

    // Create new buffer: header + mask data (as float32)
    const dataSize = maskData.length * 4;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const destBytes = new Uint8Array(buffer);
    const destView = new DataView(buffer);

    // Copy entire header from source
    destBytes.set(srcBytes.slice(0, headerSize));

    // Update datatype to FLOAT32 (16) at offset 70
    destView.setInt16(70, 16, true);
    // Update bitpix to 32 at offset 72
    destView.setInt16(72, 32, true);

    // Make it 3D (remove 4th dimension if present)
    destView.setInt16(40, 3, true);  // ndim = 3
    destView.setInt16(48, 1, true);  // dim[4] = 1

    // Copy mask data
    const dataView = new Float32Array(buffer, headerSize);
    dataView.set(maskData);

    return buffer;
  }

  // Determine which pipeline stages can be skipped based on settings changes
  determineSkipStages() {
    if (!this.pipelineHasRun || !this.lastRunSettings) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    const current = this.pipelineSettings;
    const last = this.lastRunSettings;

    // Check if unwrapping settings changed
    const unwrapChanged =
      current.unwrapMethod !== last.unwrapMethod ||
      (current.unwrapMethod === 'romeo' &&
       current.romeo.weighting !== last.romeo?.weighting);

    // Check if background removal settings changed
    const bgChanged =
      current.backgroundRemoval !== last.backgroundRemoval ||
      (current.backgroundRemoval === 'vsharp' && (
        current.vsharp.maxRadius !== last.vsharp?.maxRadius ||
        current.vsharp.minRadius !== last.vsharp?.minRadius ||
        current.vsharp.threshold !== last.vsharp?.threshold
      )) ||
      (current.backgroundRemoval === 'smv' &&
        current.smv.radius !== last.smv?.radius);

    // If unwrap changed, can't skip anything
    if (unwrapChanged) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    // If only dipole inversion changed, can skip both unwrap and bg removal
    if (!bgChanged) {
      return { skipUnwrap: true, skipBgRemoval: true };
    }

    // If bg removal changed but not unwrap, can skip unwrap only
    return { skipUnwrap: true, skipBgRemoval: false };
  }

  async runRomeoQSM() {
    // Validation
    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;
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
    const unwrapMode = document.getElementById('unwrapMode').value;

    if (!magField || magField <= 0) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    try {
      // Initialize worker if needed
      await this.initializeWorker();

      this.updateOutput("Starting ROMEO QSM Pipeline...");

      // Read file buffers
      const magnitudeBuffers = [];
      const phaseBuffers = [];

      for (let i = 0; i < magCount; i++) {
        const magFile = this.multiEchoFiles.magnitude[i]?.file;
        const phaseFile = this.multiEchoFiles.phase[i]?.file;

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
        this.updateOutput("Using custom edited mask");
      }

      // Determine which stages can be skipped based on settings changes
      const skipStages = this.determineSkipStages();
      if (skipStages.skipUnwrap) {
        this.updateOutput("Reusing cached unwrapped phase data");
      }
      if (skipStages.skipBgRemoval) {
        this.updateOutput("Reusing cached background-removed data");
      }

      // Send to worker with pipeline settings
      this.worker.postMessage({
        type: 'run',
        data: {
          magnitudeBuffers,
          phaseBuffers,
          echoTimes,
          magField,
          unwrapMode,
          maskThreshold: this.maskThreshold,
          customMaskBuffer,
          pipelineSettings: this.pipelineSettings,
          skipStages
        }
      });

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      console.error(error);
    }
  }

  showStageButtons() {
    document.getElementById('stage-buttons').classList.remove('hidden');
    this.updateDownloadButtons();
  }

  updateDownloadButtons() {
    // Enable buttons based on available data
    // This is a simplified version - in production you'd check actual data availability
    const buttons = ['downloadMagnitude', 'downloadPhase', 'downloadMask', 'downloadB0', 'downloadBgRemoved', 'downloadDipoleInversed'];
    buttons.forEach(id => {
      const button = document.getElementById(id);
      button.disabled = false; // Enable all buttons after processing
    });
  }

  async showStage(stage) {
    if (!this.worker || !this.workerReady) {
      this.updateOutput("Pipeline not yet run");
      return;
    }

    try {
      this.updateOutput(`Loading ${stage}...`);

      // Request stage data from worker
      this.worker.postMessage({
        type: 'getStage',
        data: { stage }
      });

      // Wait for response
      const result = await new Promise((resolve) => {
        this.pendingStageResolve = resolve;
      });

      // Create file from bytes
      const blob = new Blob([result.data], { type: 'application/octet-stream' });
      const file = new File([blob], `${stage}.nii`, { type: 'application/octet-stream' });

      // Load in viewer
      await this.loadAndVisualizeFile(file, result.description);

      this.results[stage] = { file: file, path: `${stage}.nii` };

    } catch (error) {
      this.updateOutput(`Error showing ${stage}: ${error.message}`);
    }
  }

  async downloadStage(stage) {
    if (!this.results[stage]?.file) {
      await this.showStage(stage); // Generate if not already available
    }
    
    if (this.results[stage]?.file) {
      const file = this.results[stage].file;
      const url = URL.createObjectURL(file);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${stage}_${new Date().toISOString().slice(0,10)}.nii`;
      a.click();
      URL.revokeObjectURL(url);
    }
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

  async runBET() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    try {
      this.updateOutput("Starting BET brain extraction...");
      this.setProgress(0.05, 'Initializing BET...');

      // Initialize worker if needed
      this.setupWorker();

      // Load magnitude file if not already cached
      if (!this.magnitudeFileBytes) {
        const file = this.multiEchoFiles.magnitude[0].file;
        this.magnitudeFileBytes = await file.arrayBuffer();
      }

      // Load magnitude into NiiVue to get dimensions
      if (!this.magnitudeVolume) {
        const file = this.multiEchoFiles.magnitude[0].file;
        const url = URL.createObjectURL(file);
        await this.nv.loadVolumes([{ url: url, name: file.name }]);
        URL.revokeObjectURL(url);

        if (this.nv.volumes.length > 0) {
          this.magnitudeVolume = this.nv.volumes[0];
          this.magnitudeData = this.magnitudeVolume.img;
          let max = -Infinity;
          for (let i = 0; i < this.magnitudeData.length; i++) {
            if (this.magnitudeData[i] > max) max = this.magnitudeData[i];
          }
          this.magnitudeMax = max;
        }
      }

      // Extract dimensions from NIfTI header
      const srcView = new DataView(this.magnitudeFileBytes);
      const nx = srcView.getInt16(42, true);
      const ny = srcView.getInt16(44, true);
      const nz = srcView.getInt16(46, true);
      this.maskDims = [nx, ny, nz];

      // Get voxel size
      const dx = srcView.getFloat32(80, true);
      const dy = srcView.getFloat32(84, true);
      const dz = srcView.getFloat32(88, true);
      const voxelSize = [dz || 1, dy || 1, dx || 1]; // z, y, x order for Python

      this.updateOutput(`Image dimensions: ${nx}x${ny}x${nz}, voxel size: ${dx.toFixed(2)}x${dy.toFixed(2)}x${dz.toFixed(2)}mm`);

      // Fetch BET code
      const betCode = await fetch('./python/bet_python.py').then(r => r.text());

      // Set up handler for BET messages
      const betHandler = (e) => {
        const { type, ...data } = e.data;

        switch (type) {
          case 'betProgress':
            this.setProgress(data.value, data.text);
            break;
          case 'betLog':
            this.updateOutput(data.message);
            break;
          case 'betComplete':
            this.worker.removeEventListener('message', betHandler);
            this.handleBETComplete(data);
            break;
          case 'betError':
            this.worker.removeEventListener('message', betHandler);
            this.updateOutput(`BET Error: ${data.message}`);
            this.setProgress(0, 'BET Failed');
            break;
        }
      };
      this.worker.addEventListener('message', betHandler);

      // Send BET request to worker with settings
      this.worker.postMessage({
        type: 'runBET',
        data: {
          magnitudeBuffer: this.magnitudeFileBytes,
          voxelSize: voxelSize,
          betCode: betCode,
          fractionalIntensity: this.betSettings.fractionalIntensity,
          iterations: this.betSettings.iterations,
          subdivisions: this.betSettings.subdivisions
        }
      });

    } catch (error) {
      this.updateOutput(`BET Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      console.error(error);
    }
  }

  async handleBETComplete(data) {
    try {
      this.updateOutput("BET completed, loading mask...");

      // Convert the mask data to Float32Array
      const maskData = new Float32Array(data.maskData);

      // Store as both current and original mask
      this.currentMaskData = maskData;
      this.originalMaskData = new Float32Array(maskData);

      // Display the mask
      await this.displayCurrentMask();

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Update run button state (mask is now available)
      this.updateEchoInfo();

      this.setProgress(1.0, 'BET Complete');
      this.updateOutput(`BET brain extraction complete. Coverage: ${data.coverage}`);
    } catch (error) {
      this.updateOutput(`Error displaying BET mask: ${error.message}`);
      console.error(error);
    }
  }

  // Pipeline Settings Modal
  openPipelineSettingsModal() {
    // Populate form with current settings

    // Unwrap method and mode
    const unwrapMethod = this.pipelineSettings.unwrapMethod || 'romeo';
    document.getElementById('unwrapMethod').value = unwrapMethod;
    document.getElementById('unwrapMode').value = this.pipelineSettings.unwrapMode || 'individual';
    document.getElementById('romeoSettings').style.display = unwrapMethod === 'romeo' ? 'block' : 'none';
    document.getElementById('laplacianSettings').style.display = unwrapMethod === 'laplacian' ? 'block' : 'none';
    document.getElementById('romeoWeighting').value = this.pipelineSettings.romeo.weighting;

    // Background removal method
    document.getElementById('bgRemovalMethod').value = this.pipelineSettings.backgroundRemoval;
    const isVsharp = this.pipelineSettings.backgroundRemoval === 'vsharp';
    document.getElementById('vsharpSettings').style.display = isVsharp ? 'block' : 'none';
    document.getElementById('smvSettings').style.display = isVsharp ? 'none' : 'block';

    // V-SHARP settings
    document.getElementById('vsharpMaxRadius').value = this.pipelineSettings.vsharp.maxRadius;
    document.getElementById('vsharpMinRadius').value = this.pipelineSettings.vsharp.minRadius;
    document.getElementById('vsharpThreshold').value = this.pipelineSettings.vsharp.threshold;

    // SMV settings
    document.getElementById('smvRadius').value = this.pipelineSettings.smv.radius;

    // Dipole inversion method
    const dipoleMethod = this.pipelineSettings.dipoleInversion;
    document.getElementById('dipoleMethod').value = dipoleMethod;
    document.getElementById('tkdSettings').style.display = dipoleMethod === 'tkd' ? 'block' : 'none';
    document.getElementById('tikhonovSettings').style.display = dipoleMethod === 'tikhonov' ? 'block' : 'none';
    document.getElementById('tvSettings').style.display = dipoleMethod === 'tv' ? 'block' : 'none';
    document.getElementById('rtsSettings').style.display = dipoleMethod === 'rts' ? 'block' : 'none';

    // TKD settings
    document.getElementById('tkdThreshold').value = this.pipelineSettings.tkd.threshold;

    // Tikhonov settings
    document.getElementById('tikhLambda').value = this.pipelineSettings.tikhonov.lambda;
    document.getElementById('tikhReg').value = this.pipelineSettings.tikhonov.reg;

    // TV-ADMM settings
    document.getElementById('tvLambda').value = this.pipelineSettings.tv.lambda;
    document.getElementById('tvMaxIter').value = this.pipelineSettings.tv.maxIter;
    document.getElementById('tvTol').value = this.pipelineSettings.tv.tol;

    // RTS settings
    document.getElementById('rtsDelta').value = this.pipelineSettings.rts.delta;
    document.getElementById('rtsMu').value = this.pipelineSettings.rts.mu;
    document.getElementById('rtsRho').value = this.pipelineSettings.rts.rho;
    document.getElementById('rtsMaxIter').value = this.pipelineSettings.rts.maxIter;

    document.getElementById('pipelineSettingsModal').classList.add('active');
  }

  closePipelineSettingsModal() {
    document.getElementById('pipelineSettingsModal').classList.remove('active');
  }

  resetPipelineSettings() {
    // Reset to defaults

    // Unwrap method and mode
    document.getElementById('unwrapMethod').value = 'romeo';
    document.getElementById('unwrapMode').value = 'individual';
    document.getElementById('romeoSettings').style.display = 'block';
    document.getElementById('laplacianSettings').style.display = 'none';
    document.getElementById('romeoWeighting').value = 'phase_snr';

    // Background removal
    document.getElementById('bgRemovalMethod').value = 'vsharp';
    document.getElementById('vsharpSettings').style.display = 'block';
    document.getElementById('smvSettings').style.display = 'none';
    document.getElementById('vsharpMaxRadius').value = 18;
    document.getElementById('vsharpMinRadius').value = 2;
    document.getElementById('vsharpThreshold').value = 0.05;
    document.getElementById('smvRadius').value = 5;

    // Dipole inversion - default to TV-ADMM
    document.getElementById('dipoleMethod').value = 'tv';
    document.getElementById('tkdSettings').style.display = 'none';
    document.getElementById('tikhonovSettings').style.display = 'none';
    document.getElementById('tvSettings').style.display = 'block';
    document.getElementById('rtsSettings').style.display = 'none';

    // TKD
    document.getElementById('tkdThreshold').value = 0.15;

    // Tikhonov
    document.getElementById('tikhLambda').value = 0.01;
    document.getElementById('tikhReg').value = 'identity';

    // TV-ADMM
    document.getElementById('tvLambda').value = 0.001;
    document.getElementById('tvMaxIter').value = 250;
    document.getElementById('tvTol').value = 0.001;

    // RTS
    document.getElementById('rtsDelta').value = 0.15;
    document.getElementById('rtsMu').value = 100000;
    document.getElementById('rtsRho').value = 10;
    document.getElementById('rtsMaxIter').value = 20;
  }

  runPipelineWithSettings() {
    // Save settings from form
    this.pipelineSettings = {
      unwrapMethod: document.getElementById('unwrapMethod').value,
      unwrapMode: document.getElementById('unwrapMode').value,
      romeo: {
        weighting: document.getElementById('romeoWeighting').value
      },
      backgroundRemoval: document.getElementById('bgRemovalMethod').value,
      vsharp: {
        maxRadius: parseFloat(document.getElementById('vsharpMaxRadius').value),
        minRadius: parseFloat(document.getElementById('vsharpMinRadius').value),
        threshold: parseFloat(document.getElementById('vsharpThreshold').value)
      },
      smv: {
        radius: parseFloat(document.getElementById('smvRadius').value)
      },
      dipoleInversion: document.getElementById('dipoleMethod').value,
      tkd: {
        threshold: parseFloat(document.getElementById('tkdThreshold').value)
      },
      tikhonov: {
        lambda: parseFloat(document.getElementById('tikhLambda').value),
        reg: document.getElementById('tikhReg').value
      },
      tv: {
        lambda: parseFloat(document.getElementById('tvLambda').value),
        maxIter: parseInt(document.getElementById('tvMaxIter').value),
        tol: parseFloat(document.getElementById('tvTol').value)
      },
      rts: {
        delta: parseFloat(document.getElementById('rtsDelta').value),
        mu: parseFloat(document.getElementById('rtsMu').value),
        rho: parseFloat(document.getElementById('rtsRho').value),
        maxIter: parseInt(document.getElementById('rtsMaxIter').value)
      }
    };

    this.closePipelineSettingsModal();
    this.runRomeoQSM();
  }

  // BET Settings Modal
  openBetSettingsModal() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    // Populate form with current settings
    document.getElementById('betFractionalIntensity').value = this.betSettings.fractionalIntensity;
    document.getElementById('betFractionalIntensityValue').textContent = this.betSettings.fractionalIntensity;
    document.getElementById('betIterations').value = this.betSettings.iterations;
    document.getElementById('betSubdivisions').value = this.betSettings.subdivisions;

    document.getElementById('betSettingsModal').classList.add('active');
  }

  closeBetSettingsModal() {
    document.getElementById('betSettingsModal').classList.remove('active');
  }

  resetBetSettings() {
    // Reset to defaults
    document.getElementById('betFractionalIntensity').value = 0.5;
    document.getElementById('betFractionalIntensityValue').textContent = '0.5';
    document.getElementById('betIterations').value = 1000;
    document.getElementById('betSubdivisions').value = 4;
  }

  runBetWithSettings() {
    // Save settings from form
    this.betSettings = {
      fractionalIntensity: parseFloat(document.getElementById('betFractionalIntensity').value),
      iterations: parseInt(document.getElementById('betIterations').value),
      subdivisions: parseInt(document.getElementById('betSubdivisions').value)
    };

    this.closeBetSettingsModal();
    this.runBET();
  }
}

// Initialize the app - this will be called after NiiVue is loaded by the module script
function initQSMApp() {
  console.log('Initializing QSM App with NiiVue:', window.Niivue);
  window.app = new QSMApp();
}

// If the script loads after DOM is ready, initialize immediately
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (window.Niivue) {
      initQSMApp();
    } else {
      console.log('Waiting for NiiVue...');
      setTimeout(() => {
        if (window.Niivue) {
          initQSMApp();
        } else {
          document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
        }
      }, 2000);
    }
  });
} else {
  // DOM already loaded
  if (window.Niivue) {
    initQSMApp();
  } else {
    setTimeout(() => {
      if (window.Niivue) {
        initQSMApp();
      } else {
        document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
      }
    }, 1000);
  }
}