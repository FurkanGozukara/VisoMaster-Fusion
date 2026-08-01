"""Manual end-to-end TensorRT provider validation with a real face image.

This is intentionally outside the automatic test suite because a missing cache can
trigger a several-minute TensorRT engine build.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.processors.models_processor import ModelsProcessor
from app.processors.workers.function_worker import FunctionWorker


class _Signal:
    def emit(self, *_args, **_kwargs) -> None:
        pass


class _VideoProcessor:
    processing = False
    current_frame_number = 0

    def stop_processing(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--provider", choices=("TensorRT", "TensorRT-Engine"), default="TensorRT"
    )
    args = parser.parse_args()

    window = SimpleNamespace(
        gpu_id=args.gpu,
        control={
            "FaceTrackingEnableToggle": False,
            "KeepModelsAliveToggle": False,
            "DetectorScoreSlider": 25,
        },
        parameters={},
        model_loaded_signal=_Signal(),
        model_loading_signal=_Signal(),
        video_processor=_VideoProcessor(),
        fixed_unet_model_name="RefLDM_UNET_EXTERNAL_KV",
    )
    processor = ModelsProcessor(window)
    function_worker = FunctionWorker(processor)
    window.function_worker = function_worker
    resolved_provider = function_worker.switch_providers_priority(args.provider)

    image_path = Path(__file__).resolve().parents[2] / "Example_Face_Video" / "face1.jpg"
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read validation image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .contiguous()
        .to(f"cuda:{args.gpu}")
    )

    started = time.perf_counter()
    boxes, landmarks, _ = function_worker.run_detect(
        image,
        detect_mode="Yunet",
        max_num=1,
        score=0.25,
        input_size=(640, 640),
        rotation_angles=[0],
    )
    torch.cuda.synchronize(args.gpu)
    session = processor.models.get("YunetN")
    engine_files = sorted(
        str(path)
        for path in Path(processor.trt_ep_options["trt_engine_cache_path"]).rglob(
            "*.engine"
        )
    )
    ok = (
        resolved_provider == args.provider
        and session is not None
        and "TensorrtExecutionProvider" in session.get_providers()
        and len(boxes) > 0
        and np.isfinite(boxes).all()
        and np.isfinite(landmarks).all()
        and bool(engine_files)
    )
    result = {
        "category": "provider",
        "name": f"{args.provider} on GPU {args.gpu}",
        "status": "PASS" if ok else "FAIL",
        "gpu": torch.cuda.get_device_name(args.gpu),
        "compute_capability": list(torch.cuda.get_device_capability(args.gpu)),
        "resolved_provider": resolved_provider,
        "session_providers": session.get_providers() if session else [],
        "provider_options": session.get_provider_options() if session else {},
        "faces": len(boxes),
        "bbox": np.asarray(boxes[0]).round(3).tolist() if len(boxes) else None,
        "engine_files": engine_files,
        "seconds": round(time.perf_counter() - started, 3),
    }
    print("VALIDATION " + json.dumps(result), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
