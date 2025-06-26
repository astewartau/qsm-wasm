class QSMApp {
  constructor() {
    this.pyodide = null;
    this.nv = new window.Niivue({ 
      onLocationChange: (data) => {
        document.getElementById("intensity").innerHTML = "&nbsp;&nbsp;" + data.string;
      }
    });
    this.currentFile = null;
    this.threshold = 75;
    this.progressBar = null;
    
    // File paths and objects
    this.files = {
      magnitude: { path: "magnitude.nii", file: null },
      phase: { path: "phase.nii", file: null },
      mask: { path: null, file: null },
      unwrapped: { path: null, file: null },
      bgRemoved: { path: null, file: null },
      final: { path: null, file: null }
    };
    
    this.init();
  }

  async init() {
    await this.setupViewer();
    this.setupProgressBar();
    this.setupEventListeners();
    this.updateDownloadButtons(); // Initialize button states
    // Don't load Pyodide until needed
    this.updateOutput("Ready to load images. Pyodide will load when you run the QSM pipeline.");
  }

  async setupViewer() {
    await this.nv.attachTo("gl1");
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
  }

  setupProgressBar() {
    this.progressBar = new ProgressBar.Line('#progressBar', {
      strokeWidth: 4,
      easing: 'easeInOut',
      duration: 300,
      color: '#ED6A5A',
      trailColor: '#eee',
      trailWidth: 1,
      from: {color: '#ED6A5A'},
      to: {color: '#A8E6A3'},
      svgStyle: { width: '100%', height: '100%' },
      step: (state, bar) => {
        bar.path.setAttribute('stroke', state.color);
      }
    });
  }

  async initPyodide() {
    this.updateOutput("Loading modules...");
    this.pyodide = await loadPyodide();
    await this.pyodide.loadPackage("micropip");

    await this.pyodide.runPythonAsync(`
      import micropip
      await micropip.install("nibabel")
    `);
    await this.pyodide.loadPackage(["numpy", "scipy"]);

    // Load Python modules
    const modules = [
      "python/masking3.py",
      "python/phase_scaling.py",
      "python/unwrap.py", 
      "python/bg_removal_sharp.py",
      "python/rts_wasm_standard.py"
    ];

    for (const module of modules) {
      const code = await fetch(module).then(r => r.text());
      await this.pyodide.runPythonAsync(code);
    }

    this.updateOutput("All modules loaded.");
  }

  setupEventListeners() {
    // File upload handlers
    document.getElementById("vis_magnitude").addEventListener("click", () => this.visualizeFile('magnitude'));
    document.getElementById("vis_phase").addEventListener("click", () => this.visualizeFile('phase'));
    
    // Main pipeline
    document.getElementById("run").addEventListener("click", () => this.runMasking());
    document.getElementById("useMask").addEventListener("click", () => this.runPipeline());
    
    // Volume switching
    document.getElementById("showMagnitude").addEventListener("click", () => this.switchVolume(this.files.magnitude.file));
    document.getElementById("showPhase").addEventListener("click", () => this.switchVolume(this.files.phase.file));
    document.getElementById("showMask").addEventListener("click", () => this.switchVolume(this.files.mask.file));
    document.getElementById("showUnwrapped").addEventListener("click", () => this.switchVolume(this.files.unwrapped.file));
    document.getElementById("showBgRemoved").addEventListener("click", () => this.switchVolume(this.files.bgRemoved.file));
    document.getElementById("showDipoleInversed").addEventListener("click", () => this.switchVolume(this.files.final.file));
    
    // Download handlers
    document.getElementById("downloadCurrent").addEventListener("click", () => this.downloadCurrentFile());
    document.getElementById("downloadImage").addEventListener("click", () => this.saveScreenshot());
    
    // Individual stage download handlers
    document.getElementById("downloadMagnitude").addEventListener("click", () => this.downloadFile(this.files.magnitude.file, "magnitude.nii"));
    document.getElementById("downloadPhase").addEventListener("click", () => this.downloadFile(this.files.phase.file, "phase.nii"));
    document.getElementById("downloadMask").addEventListener("click", () => this.downloadFile(this.files.mask.file, "mask.nii"));
    document.getElementById("downloadUnwrapped").addEventListener("click", () => this.downloadFile(this.files.unwrapped.file, "fieldmap.nii"));
    document.getElementById("downloadBgRemoved").addEventListener("click", () => this.downloadFile(this.files.bgRemoved.file, "fieldmap-local.nii"));
    document.getElementById("downloadDipoleInversed").addEventListener("click", () => this.downloadFile(this.files.final.file, "qsm-result.nii"));
    
    // Show download section when stage buttons are shown
    const stageButtonsObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'hidden') {
          const stageButtons = document.getElementById("stage-buttons");
          const downloadSection = document.getElementById("download-section");
          if (!stageButtons.hidden) {
            downloadSection.classList.remove("hidden");
          }
        }
      });
    });
    stageButtonsObserver.observe(document.getElementById("stage-buttons"), { attributes: true });
    
    // Settings upload
    document.getElementById("settingsFile").addEventListener("change", (e) => this.loadSettings(e));
  }

  async visualizeFile(type) {
    const fileInput = document.getElementById(type);
    const file = fileInput.files[0];
    
    if (!file) {
      this.updateOutput(`Please upload ${type} image.`);
      return;
    }

    this.files[type].file = file;
    // Only save to filesystem when Pyodide is loaded, not for visualization
    
    try {
      this.updateOutput(`Loading ${type} image...`);
      
      // Clear existing volumes
      if (this.nv.volumes.length > 0) {
        this.nv.removeVolumeByIndex(0);
      }
      
      // Use the correct NiiVue API to load File objects
      const image = await window.NVImage.loadFromFile({file: file});
      this.nv.addVolume(image);
      
      // Set currentFile before setupContrast
      this.currentFile = file;
      this.setupContrast(this.nv.volumes[0]);
      
      // Update download buttons since we now have magnitude/phase files
      this.updateDownloadButtons();
      
      this.updateOutput(`${type} image loaded successfully.`);
    } catch (err) {
      console.error(err);
      this.updateOutput(`Error loading ${type} image: ${err.message}`);
    }
  }

  async runMasking() {
    const magnitudeFile = document.getElementById("magnitude").files[0];
    const phaseFile = document.getElementById("phase").files[0];
    
    if (!magnitudeFile || !phaseFile) {
      this.updateOutput("Please upload both magnitude and phase images.");
      return;
    }

    // Initialize Pyodide if not already loaded
    if (!this.pyodide) {
      this.updateOutput("Loading Python environment... This may take a moment.");
      await this.initPyodide();
    }

    this.files.magnitude.file = magnitudeFile;
    this.files.phase.file = phaseFile;
    
    // Now save to Pyodide filesystem since it's loaded
    await this.saveToFS(magnitudeFile, this.files.magnitude.path);
    await this.saveToFS(phaseFile, this.files.phase.path);

    try {
      this.updateOutput("Creating initial mask...");
      const runMasking = this.pyodide.globals.get("run_masking");
      this.files.mask.path = runMasking(this.files.magnitude.path, this.threshold);

      const resultBytes = this.pyodide.FS.readFile(this.files.mask.path);
      const blob = new Blob([resultBytes], { type: "application/octet-stream" });
      this.files.mask.file = new File([blob], "mask.nii");

      // Clear existing volumes
      if (this.nv.volumes.length > 0) {
        this.nv.removeVolumeByIndex(0);
      }
      
      const maskImage = await window.NVImage.loadFromFile({file: this.files.mask.file});
      this.nv.addVolume(maskImage);
      this.currentFile = this.files.mask.file;
      
      console.log('About to setup threshold...');
      this.setupThreshold();
      console.log('About to hide initial controls...');
      this.hideInitialControls();
      console.log('About to show useMask button...');
      
      this.updateOutput("Masking result loaded. Adjust threshold as needed.");
      
      const useMaskButton = document.getElementById("useMask");
      console.log('useMask button:', useMaskButton);
      useMaskButton.classList.remove('hidden');
      useMaskButton.hidden = false;
      console.log('useMask button should now be visible');
      
      // Update download button states
      this.updateDownloadButtons();
    } catch (err) {
      console.error(err);
      this.updateOutput("Error: " + err.message);
    }
  }

  async runPipeline() {
    const useMaskButton = document.getElementById("useMask");
    const thresholdControls = document.getElementById("thresholdControls");
    useMaskButton.classList.add('hidden');
    useMaskButton.hidden = true;
    thresholdControls.classList.add('hidden');
    thresholdControls.hidden = true;
    this.showProgressBar();
    
    // Ensure Pyodide is loaded (should already be loaded from masking step)
    if (!this.pyodide) {
      this.updateProgress(5, "Loading Python environment...");
      await this.initPyodide();
    }
    
    await new Promise(r => setTimeout(r, 0)); // Allow UI update

    try {
      // Phase scaling to [-π, +π] range
      this.updateProgress(8, "Scaling phase to [-π, +π] range...");
      const runPhaseScaling = this.pyodide.globals.get("run_phase_scaling");
      const scaledPhasePath = runPhaseScaling(this.files.phase.path);
      console.log(`Phase scaled from ${this.files.phase.path} to ${scaledPhasePath}`);
      
      // Phase unwrapping (using scaled phase)
      this.updateProgress(15, "Phase Unwrapping...");
      const runUnwrap = this.pyodide.globals.get("run_unwrap");
      const echoTime = parseFloat(document.getElementById("echoTime").value);
      const magField = parseFloat(document.getElementById("magField").value);
      
      this.files.unwrapped.path = runUnwrap(scaledPhasePath, echoTime, magField);
      this.files.unwrapped.file = this.createFileFromPath(this.files.unwrapped.path, "fieldmap.nii");
      this.updateDownloadButtons();

      // Background removal
      this.updateProgress(50, "Background removal...");
      await new Promise(r => setTimeout(r, 0));
      const runBgRemoval = this.pyodide.globals.get("run_bgremoval");
      this.files.bgRemoved.path = runBgRemoval(this.files.unwrapped.path, this.files.mask.path);
      this.files.bgRemoved.file = this.createFileFromPath(this.files.bgRemoved.path, "fieldmap-local.nii");
      this.updateDownloadButtons();

      // Dipole inversion
      this.updateProgress(85, "Dipole inversion...");
      await new Promise(r => setTimeout(r, 0));
      const runInversion = this.pyodide.globals.get("run_rts");
      this.files.final.path = runInversion(this.files.bgRemoved.path, this.files.mask.path);
      this.files.final.file = this.createFileFromPath(this.files.final.path, "chimap.nii");
      this.updateDownloadButtons();
      
      this.updateProgress(100, "QSM map complete.");
      
      this.switchVolume(this.files.final.file);
      this.updateOutput("QSM result loaded.");
      
      const contrastControls = document.getElementById("contrastControls");
      const stageButtons = document.getElementById("stage-buttons");
      contrastControls.classList.remove('hidden');
      contrastControls.hidden = false;
      stageButtons.classList.remove('hidden');
      stageButtons.hidden = false;
      
    } catch (err) {
      console.error(err);
      this.updateOutput("Error: " + err.message);
    }
  }

  createFileFromPath(path, filename) {
    const resultBytes = this.pyodide.FS.readFile(path);
    const blob = new Blob([resultBytes], { type: "application/octet-stream" });
    return new File([blob], filename);
  }

  async saveToFS(file, name) {
    const exists = this.pyodide.FS.analyzePath(name).exists;
    if (exists) return;
    
    const buf = await file.arrayBuffer();
    this.pyodide.FS.writeFile(name, new Uint8Array(buf));
  }

  async switchVolume(file) {
    try {
      // Clear existing volumes
      if (this.nv.volumes.length > 0) {
        this.nv.removeVolumeByIndex(0);
      }
      this.currentFile = file;
      
      const volumeImage = await window.NVImage.loadFromFile({file: file});
      this.nv.addVolume(volumeImage);
      this.setupContrast(this.nv.volumes[0]);
    } catch (err) {
      console.error(err);
      this.updateOutput("Error loading volume: " + err.message);
    }
  }

  setupContrast(volume) {
    const contrastSlider = document.getElementById("contrastSlider");
    const minInput = document.getElementById("minInput");
    const maxInput = document.getElementById("maxInput");
    const contrastControls = document.getElementById("contrastControls");

    contrastControls.classList.remove('hidden');
    contrastControls.hidden = false;

    let startMin = volume.cal_min;
    let startMax = volume.cal_max;

    if (this.currentFile && this.currentFile.name.includes("chimap")) {
      startMin = -0.1;
      startMax = 0.1;
    }

    if (contrastSlider.noUiSlider) {
      contrastSlider.noUiSlider.destroy();
    }

    noUiSlider.create(contrastSlider, {
      start: [startMin, startMax],
      connect: true,
      tooltips: true,
      range: {
        min: volume.global_min,
        max: volume.global_max
      }
    });

    minInput.value = startMin.toFixed(2);
    maxInput.value = startMax.toFixed(2);

    contrastSlider.noUiSlider.on('update', (values) => {
      const [min, max] = values.map(parseFloat);
      volume.cal_min = min;
      volume.cal_max = max;
      this.nv.updateGLVolume();
      minInput.value = min.toFixed(2);
      maxInput.value = max.toFixed(2);
    });

    [minInput, maxInput].forEach((input) => {
      input.addEventListener("change", () => {
        const min = parseFloat(minInput.value);
        const max = parseFloat(maxInput.value);
        contrastSlider.noUiSlider.set([min, max]);
      });
    });
  }

  setupThreshold() {
    console.log('Setting up threshold controls...');
    const thresholdControls = document.getElementById("thresholdControls");
    const thresholdSlider = document.getElementById("thresholdSlider");
    const thresholdValue = document.getElementById("thresholdValue");

    console.log('Elements found:', {thresholdControls, thresholdSlider, thresholdValue});
    
    thresholdControls.classList.remove('hidden');
    thresholdControls.hidden = false;
    
    // Destroy existing slider if it exists
    if (thresholdSlider.noUiSlider) {
      thresholdSlider.noUiSlider.destroy();
    }

    noUiSlider.create(thresholdSlider, {
      start: [this.threshold],
      connect: [true, false],
      range: { min: 50, max: 90 },
      step: 1,
      tooltips: true
    });
    
    console.log('Threshold slider created successfully');

    thresholdSlider.noUiSlider.on("change", async (values) => {
      thresholdValue.textContent = "Loading...";
      
      try {
        // Ensure Pyodide is loaded (should already be loaded)
        if (!this.pyodide) {
          thresholdValue.textContent = "Error: Python not loaded";
          return;
        }
        
        const runMasking = this.pyodide.globals.get("run_masking");
        this.threshold = parseInt(values[0]);
        this.files.mask.path = runMasking(this.files.magnitude.path, this.threshold);
        
        const resultBytes = this.pyodide.FS.readFile(this.files.mask.path);
        const blob = new Blob([resultBytes], { type: "application/octet-stream" });
        this.files.mask.file = new File([blob], `mask.nii`);
        
        const maskImage = await window.NVImage.loadFromFile({file: this.files.mask.file});
      this.nv.addVolume(maskImage);
        if (this.nv.volumes.length > 1) {
          this.nv.removeVolumeByIndex(0);
        }
        thresholdValue.textContent = this.threshold.toFixed(0);
      } catch (err) {
        console.error(err);
      }
    });
  }

  showProgressBar() {
    const progressContainer = document.getElementById("progressContainer");
    progressContainer.classList.remove('hidden');
    progressContainer.hidden = false;
    this.progressBar.set(0);
  }

  updateProgress(percent, label) {
    this.progressBar.animate(percent / 100);
    document.getElementById("progressLabel").textContent = label;
  }

  updateDownloadButtons() {
    // Enable/disable download buttons based on file availability
    const buttons = [
      { id: "downloadMagnitude", file: this.files.magnitude.file },
      { id: "downloadPhase", file: this.files.phase.file },
      { id: "downloadMask", file: this.files.mask.file },
      { id: "downloadUnwrapped", file: this.files.unwrapped.file },
      { id: "downloadBgRemoved", file: this.files.bgRemoved.file },
      { id: "downloadDipoleInversed", file: this.files.final.file }
    ];

    buttons.forEach(({ id, file }) => {
      const button = document.getElementById(id);
      if (button) {
        button.disabled = !file;
        button.style.opacity = file ? "1" : "0.5";
        button.title = file ? "Download this file" : "File not yet available";
      }
    });
  }

  hideInitialControls() {
    const controls = ["vis_magnitude", "vis_phase", "contrastControls", "run"];
    controls.forEach(id => {
      const element = document.getElementById(id);
      element.classList.add('hidden');
      element.hidden = true;
    });
  }

  downloadFile(file, defaultName = null) {
    if (file) {
      const url = URL.createObjectURL(file);
      const a = document.createElement("a");
      a.href = url;
      a.download = defaultName || file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.updateOutput(`Downloaded: ${a.download}`);
    } else {
      this.updateOutput("File not available for download.");
    }
  }

  downloadCurrentFile() {
    if (this.currentFile) {
      this.downloadFile(this.currentFile);
    } else {
      this.updateOutput("No file currently displayed.");
    }
  }

  saveScreenshot() {
    if (this.currentFile) {
      const fileName = this.currentFile.name.slice(0, -4) + ".png";
      this.nv.saveScene(fileName);
    } else {
      this.nv.saveScene("screenshot.png");
    }
  }

  async loadSettings(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const settings = JSON.parse(text);

      if (settings.EchoTime !== undefined) {
        document.getElementById("echoTime").value = settings.EchoTime;
      }
      if (settings.MagneticFieldStrength !== undefined) {
        document.getElementById("magField").value = settings.MagneticFieldStrength;
      }

      this.updateOutput("Settings loaded successfully.");
    } catch (err) {
      console.error(err);
      this.updateOutput("Error loading settings: " + err.message);
    }
  }

  updateOutput(message) {
    document.getElementById("output").textContent = message;
  }
}

// Initialize the app - this will be called after NiiVue is loaded by the module script
function initQSMApp() {
  console.log('Initializing QSM App with NiiVue:', window.Niivue);
  new QSMApp();
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