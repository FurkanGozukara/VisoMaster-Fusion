"""Reproducible CUDA vs TensorRT benchmark for VisoMaster's default swap path.

The parent process launches one clean worker process per backend, then writes a
standalone JSON data file and HTML report. Video decode, audio, and output
encoding are deliberately excluded so the result isolates app frame processing.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MEDIA_ROOT = Path(r"G:\VisoMaster_Fusion_v5\Example_Face_Video")
DEFAULT_VIDEO = DEFAULT_MEDIA_ROOT / "test_video.mp4"
DEFAULT_SOURCE = DEFAULT_MEDIA_ROOT / "face1.jpg"
DEFAULT_REPORT = REPO_ROOT / "benchmark_reports" / "cuda_vs_tensorrt_rtx5090.html"


def _run_quiet(command: list[str], timeout: int = 10) -> str:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=creation_flags,
    )
    return result.stdout.strip()


def query_gpu_used_mb(gpu_id: int) -> int | None:
    output = _run_quiet(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        return int(output.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


class GpuMemorySampler:
    def __init__(self, gpu_id: int, interval_seconds: float = 0.15):
        self.gpu_id = int(gpu_id)
        self.interval_seconds = float(interval_seconds)
        self.samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="benchmark-vram-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def mark(self) -> int:
        with self._lock:
            return len(self.samples)

    def peak_since(self, sample_index: int) -> int | None:
        with self._lock:
            values = [value for _, value in self.samples[sample_index:]]
        return max(values) if values else query_gpu_used_mb(self.gpu_id)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            value = query_gpu_used_mb(self.gpu_id)
            if value is not None:
                with self._lock:
                    self.samples.append((time.time(), value))
            self._stop_event.wait(self.interval_seconds)


def _build_target_face(main_window, source_path: Path, video_path: Path):
    import cv2
    import numpy as np
    import torch
    from PySide6 import QtGui

    from app.helpers import miscellaneous as misc_helpers
    from app.ui.widgets.actions import list_view_actions

    control = main_window.control.copy()
    recognition_model = str(control["RecognitionModelSelection"])

    def recognize_largest_face(image_bgr):
        image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        image_tensor = (
            torch.from_numpy(image_rgb)
            .to(main_window.models_processor.device)
            .permute(2, 0, 1)
        )
        bboxes, kpss_5, _ = main_window.function_worker.run_detect(
            image_tensor,
            control["DetectorModelSelection"],
            max_num=int(control["MaxFacesToDetectSlider"]),
            score=float(control["DetectorScoreSlider"]) / 100.0,
            input_size=(512, 512),
            use_landmark_detection=False,
            landmark_detect_mode=control["LandmarkDetectModelSelection"],
            landmark_score=float(control["LandmarkDetectScoreSlider"]) / 100.0,
            from_points=False,
            rotation_angles=[0],
        )
        if not len(bboxes):
            raise RuntimeError("No face was detected in benchmark setup media.")
        largest_index = max(
            range(len(bboxes)),
            key=lambda index: float(
                (bboxes[index][2] - bboxes[index][0])
                * (bboxes[index][3] - bboxes[index][1])
            ),
        )
        embedding, crop_rgb = main_window.function_worker.run_recognize_direct(
            image_tensor,
            kpss_5[largest_index],
            "Auto",
            recognition_model,
        )
        if embedding is None or crop_rgb is None:
            raise RuntimeError("Face recognition failed during benchmark setup.")
        crop_bgr = np.ascontiguousarray(crop_rgb.detach().cpu().numpy()[..., ::-1])
        return embedding, crop_bgr, len(bboxes)

    source_bgr = misc_helpers.read_image_file(str(source_path))
    if source_bgr is None:
        raise RuntimeError(f"Could not read source face: {source_path}")
    source_embedding, _, _ = recognize_largest_face(source_bgr)

    capture = cv2.VideoCapture(str(video_path))
    ok, target_frame_bgr = capture.read()
    capture.release()
    if not ok or target_frame_bgr is None:
        raise RuntimeError(f"Could not read target video: {video_path}")
    target_embedding, target_crop_bgr, target_face_count = recognize_largest_face(
        target_frame_bgr
    )

    target_image = QtGui.QImage(
        target_crop_bgr.data,
        target_crop_bgr.shape[1],
        target_crop_bgr.shape[0],
        int(target_crop_bgr.strides[0]),
        QtGui.QImage.Format.Format_BGR888,
    ).copy()
    face_id = "benchmark-target"
    list_view_actions.add_media_thumbnail_to_target_faces_list(
        main_window,
        target_crop_bgr,
        {recognition_model: target_embedding},
        target_image,
        face_id,
    )
    target_face = main_window.target_faces[face_id]
    swap_model = str(main_window.parameters[face_id]["SwapModelSelection"])
    arcface_model = main_window.function_worker.get_arcface_model(swap_model)
    target_face.assigned_input_embedding[arcface_model] = source_embedding
    main_window.swapfacesButton.setChecked(True)
    main_window.editFacesButton.setChecked(False)
    return (
        face_id,
        recognition_model,
        swap_model,
        target_frame_bgr,
        target_face_count,
    )


def _read_rgb_frames(video_path: Path, frame_count: int):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame_number in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            yield frame_number, np.ascontiguousarray(frame_bgr[..., ::-1])
    finally:
        capture.release()


def run_worker(args: argparse.Namespace) -> dict:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2
    import numpy as np
    import onnxruntime
    import torch
    from PySide6 import QtWidgets

    from app.processors.workers.frame_worker import FrameWorker
    from app.ui import main_ui

    baseline_gpu_mb = query_gpu_used_mb(args.gpu_id)
    sampler = GpuMemorySampler(args.gpu_id)
    sampler.start()
    total_peak_mark = sampler.mark()

    main_ui.MainWindow.load_last_workspace = lambda self: None
    main_ui.MainWindow.closeEvent = lambda self, event: event.accept()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    initialization_started = time.perf_counter()
    main_window = main_ui.MainWindow(gpu_id=args.gpu_id)
    app.processEvents()
    initialization_seconds = time.perf_counter() - initialization_started
    app_idle_gpu_mb = query_gpu_used_mb(args.gpu_id)

    setup_started = time.perf_counter()
    resolved_provider = main_window.function_worker.switch_providers_priority(
        args.provider
    )
    main_window.control["ProvidersPrioritySelection"] = resolved_provider
    main_window.control["nThreadsSlider"] = 1
    (
        face_id,
        recognition_model,
        swap_model,
        target_frame_bgr,
        target_face_count,
    ) = _build_target_face(main_window, Path(args.source), Path(args.video))
    models_loaded_gpu_mb = query_gpu_used_mb(args.gpu_id)

    control = main_window.control.copy()
    worker = FrameWorker(
        main_window,
        frame=np.ascontiguousarray(target_frame_bgr[..., ::-1]),
        frame_number=0,
        is_single_frame=True,
    )
    worker.parameters = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in main_window.parameters.items()
    }

    warmup_frames = list(_read_rgb_frames(Path(args.video), args.warmup_frames))
    if len(warmup_frames) < args.warmup_frames:
        raise RuntimeError("Target video is too short for the requested warmup.")
    for frame_number, frame_rgb in warmup_frames:
        worker.frame_number = frame_number
        worker.frame = frame_rgb
        worker.process_frame(control, threading.Event())
    torch.cuda.synchronize(args.gpu_id)
    setup_seconds = time.perf_counter() - setup_started
    warm_gpu_mb = query_gpu_used_mb(args.gpu_id)

    processing_peak_mark = sampler.mark()
    pass_results = []
    all_latencies_ms: list[float] = []
    last_output_bgr = None
    last_input_rgb = None

    for repeat_index in range(args.repeats):
        latencies_ms = []
        processed_frames = 0
        for frame_number, frame_rgb in _read_rgb_frames(Path(args.video), args.frames):
            worker.frame_number = frame_number
            worker.frame = frame_rgb
            frame_started = time.perf_counter()
            last_output_bgr = worker.process_frame(control, threading.Event())
            torch.cuda.synchronize(args.gpu_id)
            latency_ms = (time.perf_counter() - frame_started) * 1000.0
            latencies_ms.append(latency_ms)
            all_latencies_ms.append(latency_ms)
            processed_frames += 1
            last_input_rgb = frame_rgb

        if processed_frames != args.frames:
            raise RuntimeError(
                f"Expected {args.frames} frames, processed {processed_frames}."
            )
        pass_seconds = sum(latencies_ms) / 1000.0
        pass_results.append(
            {
                "repeat": repeat_index + 1,
                "frames": processed_frames,
                "seconds": pass_seconds,
                "fps": processed_frames / pass_seconds,
                "mean_latency_ms": statistics.fmean(latencies_ms),
                "median_latency_ms": statistics.median(latencies_ms),
            }
        )

    torch.cuda.synchronize(args.gpu_id)
    steady_gpu_mb = query_gpu_used_mb(args.gpu_id)
    processing_peak_gpu_mb = sampler.peak_since(processing_peak_mark)
    total_peak_gpu_mb = sampler.peak_since(total_peak_mark)
    sampler.stop()

    if last_output_bgr is None or last_input_rgb is None:
        raise RuntimeError("Benchmark produced no frames.")
    original_bgr = np.ascontiguousarray(last_input_rgb[..., ::-1])
    mean_absolute_output_change = float(
        np.mean(
            np.abs(last_output_bgr.astype(np.int16) - original_bgr.astype(np.int16))
        )
    )

    active_sessions = {}
    for model_name, session in main_window.models_processor.models.items():
        if session is not None and hasattr(session, "get_providers"):
            active_sessions[model_name] = session.get_providers()

    capture = cv2.VideoCapture(str(args.video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    total_measured_seconds = sum(all_latencies_ms) / 1000.0
    sorted_latencies = sorted(all_latencies_ms)
    p95_index = min(
        len(sorted_latencies) - 1,
        max(0, int(round(0.95 * len(sorted_latencies))) - 1),
    )
    baseline = baseline_gpu_mb or 0
    result = {
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "gpu_id": args.gpu_id,
        "video": str(Path(args.video).resolve()),
        "source": str(Path(args.source).resolve()),
        "source_video": {
            "width": width,
            "height": height,
            "fps": source_fps,
            "frames": source_frames,
        },
        "benchmark": {
            "frames_per_repeat": args.frames,
            "repeats": args.repeats,
            "warmup_frames": args.warmup_frames,
            "total_frames": len(all_latencies_ms),
            "total_seconds": total_measured_seconds,
            "fps": len(all_latencies_ms) / total_measured_seconds,
            "mean_latency_ms": statistics.fmean(all_latencies_ms),
            "median_latency_ms": statistics.median(all_latencies_ms),
            "p95_latency_ms": sorted_latencies[p95_index],
            "pass_fps_stddev": statistics.pstdev(
                [item["fps"] for item in pass_results]
            ),
            "passes": pass_results,
        },
        "memory_mb": {
            "system_baseline": baseline_gpu_mb,
            "app_idle": app_idle_gpu_mb,
            "models_loaded": models_loaded_gpu_mb,
            "warm_processing": warm_gpu_mb,
            "steady_after": steady_gpu_mb,
            "processing_peak": processing_peak_gpu_mb,
            "total_run_peak": total_peak_gpu_mb,
            "processing_peak_increment": (
                processing_peak_gpu_mb - baseline
                if processing_peak_gpu_mb is not None
                else None
            ),
            "steady_increment": (
                steady_gpu_mb - baseline if steady_gpu_mb is not None else None
            ),
        },
        "setup": {
            "app_initialization_seconds": initialization_seconds,
            "backend_model_setup_and_warmup_seconds": setup_seconds,
        },
        "pipeline": {
            "recognition_model": recognition_model,
            "swap_model": swap_model,
            "target_face_id": face_id,
            "target_faces_in_first_frame": target_face_count,
            "mean_absolute_output_change": mean_absolute_output_change,
        },
        "active_sessions": active_sessions,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    try:
        import tensorrt

        result["versions"]["tensorrt"] = tensorrt.__version__
    except ImportError:
        result["versions"]["tensorrt"] = None

    main_window.hide()
    Path(args.worker_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _gpu_metadata(gpu_id: int) -> dict:
    output = _run_quiet(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    parts = [part.strip() for part in output.split(",")]
    return {
        "name": parts[0] if parts else "Unknown GPU",
        "memory_total_mb": int(parts[1]) if len(parts) > 1 else None,
        "driver_version": parts[2] if len(parts) > 2 else "Unknown",
    }


def _format_number(value, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{decimals}f}{suffix}"


def build_html_report(payload: dict) -> str:
    cuda = payload["results"]["CUDA"]
    tensorrt = payload["results"]["TensorRT"]
    cuda_fps = float(cuda["benchmark"]["fps"])
    trt_fps = float(tensorrt["benchmark"]["fps"])
    speedup = trt_fps / cuda_fps if cuda_fps else 0.0
    speed_percent = (speedup - 1.0) * 100.0
    cuda_vram = cuda["memory_mb"]["processing_peak_increment"]
    trt_vram = tensorrt["memory_mb"]["processing_peak_increment"]
    vram_delta = (
        trt_vram - cuda_vram if cuda_vram is not None and trt_vram is not None else None
    )
    max_fps = max(cuda_fps, trt_fps, 1.0)
    max_vram = max(cuda_vram or 0, trt_vram or 0, 1)
    source_fps = float(cuda["source_video"]["fps"])
    faster_name = "TensorRT" if trt_fps >= cuda_fps else "CUDA"
    slower_name = "CUDA" if faster_name == "TensorRT" else "TensorRT"
    delta_word = "faster" if speed_percent >= 0 else "slower"
    vram_word = "more" if (vram_delta or 0) >= 0 else "less"
    generated = html.escape(payload["generated_at"])
    gpu = payload["system"]["gpu"]
    revision_label = html.escape(payload["repository_revision"])
    if payload.get("working_tree_dirty"):
        revision_label += " + local changes"

    def pass_rows(provider_name: str, result: dict) -> str:
        return "".join(
            "<tr>"
            f"<td>{html.escape(provider_name)}</td>"
            f"<td>{item['repeat']}</td>"
            f"<td>{item['frames']}</td>"
            f"<td>{item['fps']:.2f}</td>"
            f"<td>{item['mean_latency_ms']:.2f} ms</td>"
            f"<td>{item['median_latency_ms']:.2f} ms</td>"
            "</tr>"
            for item in result["benchmark"]["passes"]
        )

    recommendation = (
        f"Use {faster_name} when throughput is the priority on this machine. "
        f"It led {slower_name} in this controlled default-swap workload."
    )
    if abs(speed_percent) < 5:
        recommendation = (
            "The measured throughput is effectively tied. Prefer CUDA for simpler "
            "startup and portability unless TensorRT helps a different enabled model."
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUDA vs TensorRT | VisoMaster Backend Benchmark</title>
  <style>
    :root {{ color-scheme: dark; --bg:#111416; --surface:#191d20; --line:#343b40; --text:#edf1f2; --muted:#aab4b8; --cuda:#39b8c7; --trt:#f0a84b; --good:#71c587; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.45; letter-spacing:0; }}
    header {{ border-bottom:1px solid var(--line); background:#15191b; }}
    .wrap {{ width:min(1120px, calc(100% - 32px)); margin:0 auto; }}
    header .wrap {{ padding:32px 0 26px; }}
    h1 {{ margin:0 0 8px; font-size:42px; font-weight:760; letter-spacing:0; }}
    h2 {{ margin:0 0 16px; font-size:21px; letter-spacing:0; }}
    p {{ margin:0; }}
    .lede {{ color:var(--muted); max-width:760px; font-size:16px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:18px; color:var(--muted); font-size:13px; }}
    main {{ padding:26px 0 44px; }}
    section {{ padding:24px 0; border-bottom:1px solid var(--line); }}
    .result-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .metric {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px; min-width:0; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; font-weight:700; }}
    .metric strong {{ display:block; margin-top:5px; font-size:32px; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
    .metric small {{ color:var(--muted); }}
    .accent {{ color:var(--good); }}
    .chart-row {{ display:grid; grid-template-columns:92px 1fr 100px; gap:12px; align-items:center; margin:12px 0; }}
    .track {{ height:20px; background:#252b2f; border:1px solid var(--line); border-radius:3px; overflow:hidden; }}
    .bar {{ height:100%; min-width:2px; }}
    .bar.cuda {{ background:var(--cuda); }}
    .bar.trt {{ background:var(--trt); }}
    .value {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .legend {{ display:flex; gap:18px; color:var(--muted); font-size:13px; margin-bottom:14px; }}
    .dot {{ width:10px; height:10px; display:inline-block; margin-right:6px; border-radius:2px; }}
    .note {{ border-left:3px solid var(--good); padding:12px 14px; background:#18201b; color:#dfe9e1; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
    table {{ width:100%; border-collapse:collapse; min-width:650px; font-size:14px; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
    th:first-child,td:first-child {{ text-align:left; }}
    thead {{ background:#20262a; color:#d9e0e2; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    dl {{ display:grid; grid-template-columns:minmax(180px,280px) 1fr; gap:9px 18px; margin:0; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
    code {{ color:#d6e8ec; font-family:Consolas,monospace; font-size:12px; }}
    .fine {{ color:var(--muted); font-size:13px; margin-top:12px; }}
    @media (max-width:720px) {{ h1 {{ font-size:32px; }} .result-grid {{ grid-template-columns:1fr; }} .metric strong {{ font-size:28px; }} .chart-row {{ grid-template-columns:72px 1fr 82px; gap:8px; }} dl {{ grid-template-columns:1fr; gap:2px; }} dd {{ margin-bottom:10px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>CUDA vs TensorRT</h1>
      <p class="lede">A measured VisoMaster default face-swap pipeline comparison using the supplied 1080p example video and source face.</p>
      <div class="meta"><span>{html.escape(gpu["name"])}</span><span>{gpu["memory_total_mb"]:,} MB VRAM</span><span>Driver {html.escape(gpu["driver_version"])}</span><span>Build {revision_label}</span><span>{generated}</span></div>
    </div>
  </header>
  <main class="wrap">
    <section>
      <div class="result-grid">
        <div class="metric"><span>CUDA processing</span><strong>{cuda_fps:.2f} FPS</strong><small>{cuda["benchmark"]["median_latency_ms"]:.2f} ms median</small></div>
        <div class="metric"><span>TensorRT processing</span><strong>{trt_fps:.2f} FPS</strong><small>{tensorrt["benchmark"]["median_latency_ms"]:.2f} ms median</small></div>
        <div class="metric"><span>TensorRT difference</span><strong class="accent">{speedup:.2f}x</strong><small>{abs(speed_percent):.1f}% {delta_word} than CUDA</small></div>
      </div>
    </section>
    <section>
      <h2>Processing Speed</h2>
      <div class="legend"><span><i class="dot" style="background:var(--cuda)"></i>CUDA</span><span><i class="dot" style="background:var(--trt)"></i>TensorRT</span></div>
      <div class="chart-row"><b>CUDA</b><div class="track"><div class="bar cuda" style="width:{cuda_fps / max_fps * 100:.1f}%"></div></div><div class="value">{cuda_fps:.2f} FPS</div></div>
      <div class="chart-row"><b>TensorRT</b><div class="track"><div class="bar trt" style="width:{trt_fps / max_fps * 100:.1f}%"></div></div><div class="value">{trt_fps:.2f} FPS</div></div>
      <p class="fine">Source video rate: {source_fps:.3f} FPS. The benchmark returns every processed frame to CPU memory, matching the app pipeline boundary.</p>
    </section>
    <section>
      <h2>VRAM During Processing</h2>
      <div class="chart-row"><b>CUDA</b><div class="track"><div class="bar cuda" style="width:{(cuda_vram or 0) / max_vram * 100:.1f}%"></div></div><div class="value">{_format_number(cuda_vram, 0, " MB")}</div></div>
      <div class="chart-row"><b>TensorRT</b><div class="track"><div class="bar trt" style="width:{(trt_vram or 0) / max_vram * 100:.1f}%"></div></div><div class="value">{_format_number(trt_vram, 0, " MB")}</div></div>
      <p class="fine">TensorRT used {_format_number(abs(vram_delta) if vram_delta is not None else None, 0, " MB")} {vram_word} peak VRAM than CUDA. Values are GPU-wide used-memory increments above the idle baseline because Windows WDDM does not expose per-process VRAM through <code>nvidia-smi</code>.</p>
    </section>
    <section>
      <h2>Recommendation</h2>
      <p class="note">{html.escape(recommendation)}</p>
    </section>
    <section>
      <h2>Repeat Results</h2>
      <div class="table-wrap"><table><thead><tr><th>Backend</th><th>Run</th><th>Frames</th><th>FPS</th><th>Mean latency</th><th>Median latency</th></tr></thead><tbody>{pass_rows("CUDA", cuda)}{pass_rows("TensorRT", tensorrt)}</tbody></table></div>
    </section>
    <section>
      <h2>Method</h2>
      <dl>
        <dt>Workload</dt><dd>RetinaFace detection, Inswapper128ArcFace recognition and matching, one Inswapper128 swap, mask/paste, and GPU-to-CPU frame return.</dd>
        <dt>Media</dt><dd><code>{html.escape(cuda["video"])}</code><br><code>{html.escape(cuda["source"])}</code></dd>
        <dt>Video</dt><dd>{cuda["source_video"]["width"]}x{cuda["source_video"]["height"]}, {source_fps:.3f} FPS, {cuda["source_video"]["frames"]} source frames</dd>
        <dt>Measured sample</dt><dd>{cuda["benchmark"]["repeats"]} repeats x {cuda["benchmark"]["frames_per_repeat"]} sequential frames = {cuda["benchmark"]["total_frames"]} frames per backend, after {cuda["benchmark"]["warmup_frames"]} warmup frames</dd>
        <dt>Timing boundary</dt><dd>Frame processing only. File decode, audio, output encoding, and disk writes excluded.</dd>
        <dt>Concurrency</dt><dd>One controlled processing worker. Multi-worker playback or recording can change absolute FPS and VRAM, but not the models or backend path compared here.</dd>
        <dt>CUDA setup + warmup</dt><dd>{cuda["setup"]["backend_model_setup_and_warmup_seconds"]:.2f} seconds</dd>
        <dt>TensorRT setup + warmup</dt><dd>{tensorrt["setup"]["backend_model_setup_and_warmup_seconds"]:.2f} seconds, using engine/timing caches where available</dd>
        <dt>Software</dt><dd>PyTorch {html.escape(cuda["versions"]["torch"])}; ONNX Runtime {html.escape(cuda["versions"]["onnxruntime"])}; TensorRT {html.escape(str(tensorrt["versions"]["tensorrt"]))}</dd>
      </dl>
      <p class="fine">Measured results are specific to this GPU, driver, models, settings, media, engine cache state, and app build. Re-run <code>tools/benchmark_cuda_vs_tensorrt.py</code> after meaningful driver or model changes.</p>
    </section>
  </main>
</body>
</html>
"""


def run_comparison(args: argparse.Namespace) -> tuple[Path, Path]:
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO_ROOT / "temp_files" / "backend_benchmark"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "CUDA": raw_dir / "cuda_benchmark_raw.json",
        "TensorRT": raw_dir / "tensorrt_benchmark_raw.json",
    }
    results = {}
    for provider in ("CUDA", "TensorRT"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--provider",
            provider,
            "--gpu-id",
            str(args.gpu_id),
            "--video",
            str(Path(args.video).resolve()),
            "--source",
            str(Path(args.source).resolve()),
            "--frames",
            str(args.frames),
            "--repeats",
            str(args.repeats),
            "--warmup-frames",
            str(args.warmup_frames),
            "--worker-output",
            str(raw_paths[provider]),
        ]
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        print(f"[BENCH] Running {provider} benchmark...", flush=True)
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        results[provider] = json.loads(raw_paths[provider].read_text(encoding="utf-8"))

    revision = _run_quiet(["git", "rev-parse", "--short", "HEAD"])
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_revision": revision or "unknown",
        "working_tree_dirty": bool(_run_quiet(["git", "status", "--porcelain"])),
        "system": {
            "platform": platform.platform(),
            "gpu": _gpu_metadata(args.gpu_id),
        },
        "results": results,
    }
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(build_html_report(payload), encoding="utf-8")
    return report_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=("CUDA", "TensorRT"), default="CUDA")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--worker-output",
        type=Path,
        default=REPO_ROOT / "temp_files" / "backend_benchmark_worker.json",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.video).is_file():
        raise FileNotFoundError(args.video)
    if not Path(args.source).is_file():
        raise FileNotFoundError(args.source)
    if args.frames < 1 or args.repeats < 1 or args.warmup_frames < 1:
        raise ValueError("frames, repeats, and warmup-frames must be positive")

    if args.worker:
        result = run_worker(args)
        print(
            f"[BENCH] {result['provider_resolved']}: "
            f"{result['benchmark']['fps']:.2f} FPS, "
            f"{result['memory_mb']['processing_peak_increment']} MB peak increment",
            flush=True,
        )
        return 0

    report_path, json_path = run_comparison(args)
    print(f"[BENCH] HTML report: {report_path}")
    print(f"[BENCH] JSON data: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
