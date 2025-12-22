# VisoMaster Fusion

# 1-Click installers at : https://www.patreon.com/posts/121570322

### Visomaster Fusion is a powerful yet easy-to-use tool for face swapping and editing in images and videos. It utilizes AI to produce natural-looking results with minimal effort, making it ideal for both casual users and professionals.

***

# VisoMaster Fusion & Automatic Installer
**The Most Advanced 0-Shot Face Swap / Deep Fake APP - State of the Art**

**Author:** [SECourses](https://www.patreon.com/SECourses)  
**Latest Update:** 22 December 2025  
**Platform:** Windows (Portable) & Massed Compute (Cloud)

## 📂 Latest Downloads
*   **Recommended (Newer):** `VisoMaster_Fusion_v2.zip`
*   **Classic Version:** `VisoMaster_v8.zip`

*(Note: Extract and overwrite previous files if updating)*

---

## 📅 Update Log & Critical Notes

### 22 December 2025 Update
*   **New Downloader:** `Windows_Fix_Resume_Model_Download.bat` added. It will auto-hash verify and ultra-fast download all **64 models** used by VisoMaster.
*   **Portable Launcher:** `Windows_Start_Portable.bat` now uses the ultra-advanced, SHA256 verified downloader (replacing the old undependable one).
*   **Corrupt Model Fixes:** Fixed `refldm.ckpt` in the repo. The following models are now included in the auto-downloader:
    *   `refldm.ckpt` (Used at denoiser)
    *   `vqgan.ckpt`
    *   `ref_ldm_vae_decoder.onnx`
    *   `ref_ldm_vae_encoder.onnx`
    *   `vgg_combo_relu3_3_relu3_1.onnx`

### 18 December 2025 Update (Fusion Launch)
*   **Portable Mode:** Added `Windows_Start_Portable.bat`. Installs everything into a `portable-files` folder. **No Python or libraries required** on the host machine.
*   **GPU Support:** Supports **All GPUs** from RTX 1000 series to 5000 series.
*   **Environment:** Installs with **Torch 2.9** and **CUDA 12.8**.
*   **Default Device:** Set to **CUDA** by default (works faster/better than TensorRT).
*   **Speed:** Model downloader is now ~10x faster using `uv` package installing with pip.

### 24 September 2025 Update
*   **Cloud Support:** Supports Cloud GPUs (e.g., Massed Compute) with Torch 2.8, CUDA 12.9, and cuDNN 9.12.
*   **Comparison:** Developer recommends FaceFusion as #1, but VisoMaster Fusion provides a different, feature-rich desktop experience.

---

## 📊 Major Features: VisoMaster-Fusion

### 1. Job Manager & Batch Processing
*   **Complete Management:** Save, load, and manage multiple processing jobs.
*   **Automated Batch:** Queue multiple jobs and process automatically (Background processing supported).
*   **Workspace Persistence:** Saves entire state (models, settings, faces, markers).
*   **Validation:** Checks file existence and data integrity before loading.
*   **Master Assets:** Share common assets across multiple jobs.
*   **Custom Naming:** Use job names for output files.
*   **Segmented Recording:** Set multiple start/end markers for video sections and combine output.

### 2. Advanced Embedding Editor
*   **Standalone Editor:** Drag-and-drop interface.
*   **Formats:** Load/save/convert `.npy`, `.npz`, `.pkl`, `.safetensors`.
*   **Visuals:** Thumbnail previews, natural sorting, search/filter by name.
*   **Edit Tools:** Batch operations (select multiple, copy/paste, delete), Context menu (rename), Undo/Redo system.

### 3. VR180 Video Support
*   **Formats:** Hemispherical VR processing (VR180).
*   **Converters:** `Equirec2Perspec_vr.py` and `Perspec2Equirec_vr.py` included in `vr_utils.py`.
*   **Processing:** Separate Left/Right eye processing.
*   **Stitching:** Advanced feathering and mask blending for seamless results.

### 4. ReF-LDM Denoiser (Reference Latent Diffusion)
*   **Core:** Extracts Key/Value tensors from reference images (with color matching support).
*   **Modes:**
    *   **Single Step (Fast):** Configurable timestep.
    *   **Full Restore (DDIM):** High-quality restoration with multiple steps.
*   **Pipeline Position:** Apply Before Restorers, After First Restorer, or After All Restorers.
*   **Controls:** Exclusive Reference Path toggle, Base Seed, CFG Scale, Timestep, DDIM steps.

### 5. Preset System
*   **Dual Files:** Separate parameter files and control files (`_ctl.json`).
*   **UI:** Preset List Widget (browser), context menu to rename/delete, double-click to apply.

### 6. Enhanced Models
*   **Face Swappers:**
    *   `InStyleSwapper256` Version A, B, and C (Support 512 resolution).
    *   Auto Resolution Detection for `Inswapper128`.
*   **Restorer:** `GFPGAN-1024` (High-res 1024px restoration).
*   **Texture:** `VGG_Combo_relu3` (Combined VGG for differencing/texture transfer).

### 7. Advanced UI & Themes
*   **10 Themes:** True-Dark, Solarized-Dark, Solarized-Light, Dracula, Nord, Gruvbox, Monokai, etc.
*   **Panels:** Independent hide/show toggles for 'Target Videos', 'Input Faces', 'Jobs'.
*   **File Management:** "Send to Trash" (via `send2trash`), "Open File Location", Filter tracking.
*   **Help:** Integrated Help Menu with two separate documentation panels.

### 8. Advanced Launcher
*   **Tools:** Dedicated UI, built-in Git operations, automatic environment setup.
*   **Config:** `cfgtools.py` for settings management.

### 9. Face Processing
*   **XSeg:** Advanced mouth area masking, integration with DFL XSeg.
*   **Expression Restorer:** Separate Eye/Lip controls, "Normalize Eyes" (prevents fish-eye), Relative Position toggle.
*   **Face Parser:** Can run at pipeline end; "Mouth Inside" toggle.
*   **Face Restorer:** Auto-adjust blend values, Sharpness auto-adjust, **Face Restorer 2** (secondary pass).
*   **Texture Transfer:** Advanced mapping with mask manipulation.
*   **Mask View:** Real-time preview for Swap Mask, Differencing, and Texture Transfer.

### 10. Video Processing
*   **Audio:** Playback during preview, Volume slider, Delay slider (sync).
*   **Playback:** Buffering and Loop options.
*   **Recording:**
    *   Frame resize to standard 1920x1080.
    *   HDR Encoding (CPU).
    *   Direct FFMPEG parameter access.
    *   No speed limit; UI remains active during recording.
    *   Auto-open output folder.

### 11. Advanced Swap
*   **Control:** "Swap Input Face Once" (prevents false swaps on similar faces).
*   **Sharpness:** Pre-swap sharpness slider.
*   **Color Transfer:** Improved AutoColor, Secondary Pass at pipeline end, `Test_Mask` feature.
*   **Face Editor:** Selectable pipeline position, Async CUDA threads.

### 12. Performance & Workflow
*   **Optimization:** GPU used for detection (not CPU), Async CUDA threads, Optimized memory loading/unloading (VRAM management).
*   **Stability:** Separate ONNX Probe to prevent Runtime Engine Cache crashes.
*   **Workflow:** Auto-save workspace, Auto-load last workspace, Window state persistence.
*   **Experimental:** Dedicated section for testing features (e.g., MPEG Compression Simulation).
*   **Image Batch:** Automated processing for image collections.

---

## 💻 System Requirements

*   **OS:** Windows 10 / 11
*   **GPU:** NVIDIA RTX 1000, 2000, 3000, 4000, or 5000 series.
*   **Base Dependencies (For Manual Install):**
    *   Python 3.11
    *   FFmpeg
    *   CUDA 12.9
    *   cuDNN 9.12
    *   C++ Tools & MSVC
    *   Git

**Requirements Tutorial:** [View on Patreon](https://www.patreon.com/posts/111553210) | [Video Guide](https://youtu.be/DrhUHnYfwC0)

---

## 🔧 Installation Instructions

### Method 1: Windows Portable (Recommended)
*Does not require Python or libraries installed on your PC.*
1.  Download `VisoMaster_Fusion_v2.zip`.
2.  Extract to a **short path** (e.g., `C:/viso_master_v1`).
    *   *Warning: Do not use spaces in folder names.*
3.  Double-click **`Windows_Start_Portable.bat`**.
    *   *Note: Do not run as administrator.*

### Method 2: Windows Standard Install
1.  Ensure Requirements (Python 3.11, CUDA, Git, etc.) are installed.
2.  Extract zip to a short path.
3.  Run **`Windows_Install.bat`**.
4.  Run **`Windows_Start_App.bat`** to launch.

### Method 3: Massed Compute (Cloud)
*Note: RunPod is NOT supported as this app requires a Desktop GUI.*
1.  Register: [Massed Compute Signup](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute).
2.  Coupon Code: `SECourses`.
3.  Select the **SECourses** image from the Creator dropdown.
4.  Follow the file `Massed_Compute_Instructions_READ.txt` included in the zip.
5.  Supported GPUs: L40S, RTX A6000 ADA, RTX 6000 PRO.

---

## ℹ️ Important Notes & Troubleshooting

*   **TensorRT vs CUDA:** CUDA is set as the default device. While TensorRT is supported, it takes 10-15 minutes to compile every time. CUDA is recommended for speed.
*   **Path Length:** Long directory paths or spaces in folder names will cause issues. Use a short root path (e.g., `C:/VM`).
*   **Model Downloads:** If downloads fail, run `Windows_Fix_Resume_Model_Download.bat`.
*   **Browser vs Desktop:** This is a desktop GUI application, not browser-based (Gradio/Streamlit), which is why it requires specific cloud setups like Massed Compute Desktop instances.

## 🔗 Community & Support
*   **Discord:** [SECourses Discord](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388) (Ask for rank)
*   **GitHub:** [Stable Diffusion & Generative AI Repo](https://github.com/FurkanGozukara/Stable-Diffusion)
*   **Original VisoMaster Repo:** [VisoMaster GitHub](https://github.com/visomaster/VisoMaster)
*   **Social:** [Reddit](https://www.reddit.com/r/SECourses/) | [LinkedIn](https://www.linkedin.com/in/furkangozukara/)

<img width="3827" height="1938" alt="13" src="https://github.com/user-attachments/assets/32f5b9e8-78e0-4428-926f-6c1f85b9d6bd" />
<img width="2000" height="1699" alt="12" src="https://github.com/user-attachments/assets/07aa93d0-5db8-4ca4-b3df-8d3764f4c76f" />
<img width="2000" height="1575" alt="11" src="https://github.com/user-attachments/assets/4ab2a081-d8af-47d0-aaca-34d78f53d0d0" />
<img width="3834" height="1989" alt="10" src="https://github.com/user-attachments/assets/19920015-5367-446f-a854-514a54c62576" />
<img width="2589" height="1810" alt="9" src="https://github.com/user-attachments/assets/0a8094dc-bc14-4e3b-a9ca-afeacddbddda" />
<img width="3835" height="1943" alt="8" src="https://github.com/user-attachments/assets/19b9b2e8-21b4-4b97-b186-d88b76d72092" />
<img width="3839" height="1946" alt="7" src="https://github.com/user-attachments/assets/6a56dfeb-c918-466f-a31b-850895012c19" />
<img width="3830" height="1923" alt="6" src="https://github.com/user-attachments/assets/c14dbee2-1c25-428e-8726-4199e212514c" />
<img width="3824" height="1950" alt="5" src="https://github.com/user-attachments/assets/25bb0edd-cb4d-4abe-9a51-eb0c35c74c55" />
<img width="3830" height="1953" alt="4" src="https://github.com/user-attachments/assets/63318dc3-50e6-4765-a80a-afbde048ef9a" />
<img width="3814" height="1966" alt="3" src="https://github.com/user-attachments/assets/dc375dfa-a80e-462e-af9c-21cf145028df" />
<img width="2000" height="1728" alt="2" src="https://github.com/user-attachments/assets/7b990c82-d13b-42e3-9689-2338ccb1c98b" />
<img width="3808" height="1943" alt="1" src="https://github.com/user-attachments/assets/13a0e038-e641-4d06-8ae5-251896c240c7" />
