from typing import TYPE_CHECKING, Dict
from pathlib import Path
import uuid

import numpy
import cv2
import torch
import gc
from torchvision.transforms import v2
from PySide6 import QtCore, QtGui, QtWidgets

import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.face_scan_dialog import FaceScanReviewDialog
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _face_crop_to_qimage(cropped_face: numpy.ndarray) -> QtGui.QImage:
    face_img = numpy.ascontiguousarray(cropped_face.astype("uint8", copy=False))
    height, width, _channel = face_img.shape
    return QtGui.QImage(
        face_img.data,
        width,
        height,
        int(face_img.strides[0]),
        QtGui.QImage.Format.Format_BGR888,
    ).copy()


def _assign_checked_sources_to_target_face(
    main_window: "MainWindow", target_face
) -> None:
    if not (
        main_window.control.get("KeepInputToggle", False)
        or main_window.control.get("AutoSwapToggle", False)
    ):
        return

    for input_face_id, input_face_button in main_window.input_faces.items():
        if input_face_button.isChecked():
            target_face.assigned_input_faces[input_face_id] = (
                input_face_button.embedding_store
            )
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        if embed_button.isChecked():
            target_face.assigned_merged_embeddings[embedding_id] = (
                embed_button.embedding_store
            )
    target_face.calculate_assigned_input_embedding()


def clear_target_faces(main_window: "MainWindow", refresh_frame=True):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "clear target faces"
    ):
        return

    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()
    main_window.targetFacesList.clear()

    for target_face in list(main_window.target_faces.values()):
        if hasattr(target_face, "embedding_store"):
            target_face.embedding_store.clear()
        if hasattr(target_face, "assigned_input_embedding"):
            target_face.assigned_input_embedding.clear()
        if hasattr(target_face, "assigned_input_faces"):
            target_face.assigned_input_faces.clear()
        if hasattr(target_face, "assigned_merged_embeddings"):
            target_face.assigned_merged_embeddings.clear()
        if hasattr(target_face, "aged_input_embedding"):
            target_face.aged_input_embedding.clear()
        if hasattr(target_face, "aged_kv_map"):
            target_face.aged_kv_map = None
        target_face.deleteLater()
    main_window.target_faces.clear()
    main_window.parameters.clear()
    if hasattr(main_window, "issue_frames_by_face"):
        main_window.issue_frames_by_face.clear()
    if hasattr(main_window, "issue_frames"):
        main_window.issue_frames.clear()
    if hasattr(main_window, "videoSeekSlider"):
        main_window.videoSeekSlider.issue_markers = set()
        main_window.videoSeekSlider.issue_markers_sorted = []
        main_window.videoSeekSlider.update()

    main_window.selected_target_face_id = None
    # Set Parameter widget values to default
    common_widget_actions.set_widgets_values_using_face_id_parameters(
        main_window=main_window, face_id=None
    )
    video_control_actions.update_scan_review_button_states(main_window)

    # Force VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- DIRTY FLAG : CLEAR TARGETS ---
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        main_window.video_processor.ui_state_is_dirty = True

    if refresh_frame:
        common_widget_actions.refresh_frame(main_window=main_window)


def clear_input_faces(main_window: "MainWindow"):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "clear input faces"
    ):
        return

    main_window.inputFacesList.clear()

    for input_face in list(main_window.input_faces.values()):
        if hasattr(input_face, "embedding_store"):
            input_face.embedding_store.clear()
        if hasattr(input_face, "cropped_face"):
            input_face.cropped_face = None
        input_face.deleteLater()

    main_window.input_faces.clear()

    for target_face in main_window.target_faces.values():
        target_face.assigned_input_faces = {}
        target_face.calculate_assigned_input_embedding()

    # Force VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- DIRTY FLAG : CLEAR INPUTS ---
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        main_window.video_processor.ui_state_is_dirty = True

    common_widget_actions.refresh_frame(main_window=main_window)


def clear_merged_embeddings(main_window: "MainWindow"):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "clear merged embeddings"
    ):
        return

    main_window.inputEmbeddingsList.clear()

    for embed_button in list(main_window.merged_embeddings.values()):
        if hasattr(embed_button, "embedding_store"):
            embed_button.embedding_store.clear()
        if hasattr(embed_button, "kv_map"):
            embed_button.kv_map = None
        embed_button.deleteLater()

    main_window.merged_embeddings.clear()

    for target_face in main_window.target_faces.values():
        target_face.assigned_merged_embeddings = {}
        target_face.calculate_assigned_input_embedding()

    # Force VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- DIRTY FLAG : CLEAR MERGED EMBEDDINGS ---
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        main_window.video_processor.ui_state_is_dirty = True

    common_widget_actions.refresh_frame(main_window=main_window)


def uncheck_all_input_faces(main_window: "MainWindow"):
    # Uncheck All other input faces
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setChecked(False)

    # Force Garbage Collection for dangling merged tensors
    gc.collect()
    # Force PyTorch to release cached VRAM back to the OS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def uncheck_all_merged_embeddings(main_window: "MainWindow"):
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setChecked(False)

    # Force Garbage Collection for dangling merged tensors
    gc.collect()
    # Force PyTorch to release cached VRAM back to the OS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_target_faces(main_window: "MainWindow"):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(main_window, "find faces"):
        return

    control = main_window.control.copy()
    video_processor = main_window.video_processor
    if video_processor.media_path:
        frame = None
        media_capture = video_processor.media_capture

        if video_processor.file_type == "image":
            frame = misc_helpers.read_image_file(video_processor.media_path)
        elif video_processor.file_type == "video" and media_capture:
            # Position frame before read
            media_capture.set(
                cv2.CAP_PROP_POS_FRAMES, video_processor.current_frame_number
            )
            # Pass rotation
            ret, frame = misc_helpers.read_frame(
                media_capture, video_processor.media_rotation
            )
        elif video_processor.file_type == "webcam" and media_capture:
            # Pass 0 for webcam rotation
            ret, frame = misc_helpers.read_frame(media_capture, 0)

        if frame is not None:
            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            img = torch.from_numpy(frame.astype("uint8")).to(
                main_window.models_processor.device
            )
            img = img.permute(2, 0, 1)
            if control.get("ManualRotationEnableToggle", False):
                img = v2.functional.rotate(
                    img,
                    angle=control.get("ManualRotationAngleSlider", 0),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    expand=True,
                )

            _, kpss_5, _ = main_window.models_processor.run_detect(
                img,
                control.get("DetectorModelSelection", "retinaface_10g"),
                max_num=control.get("MaxFacesToDetectSlider", 1),
                score=float(control.get("DetectorScoreSlider", 50)) / 100.0,
                input_size=(512, 512),
                use_landmark_detection=control.get("LandmarkDetectToggle", False),
                landmark_detect_mode=control.get(
                    "LandmarkDetectModelSelection", "2D106Det"
                ),
                landmark_score=float(control.get("LandmarkDetectScoreSlider", 50))
                / 100.0,
                from_points=control.get("DetectFromPointsToggle", False),
                rotation_angles=[0]
                if not control.get("AutoRotationToggle", False)
                else [0, 90, 180, 270],
            )

            faces_list: list = []
            similarity_type = str("Auto")
            for face_kps in kpss_5:
                face_emb, cropped_img = (
                    main_window.models_processor.run_recognize_direct(
                        img,
                        face_kps,
                        similarity_type,
                        control.get("RecognitionModelSelection", "arcface_128"),
                    )
                )
                # Recognition can fail cleanly when a model cannot be loaded (for
                # example, transient GPU-memory pressure from another process).
                # Do not turn that recoverable condition into an AttributeError
                # below when the target-face thumbnail is constructed.
                if (
                    face_emb is None
                    or not isinstance(face_emb, numpy.ndarray)
                    or face_emb.size == 0
                    or cropped_img is None
                ):
                    continue
                faces_list.append([face_kps, face_emb, cropped_img, img])

            if faces_list:
                # Loop through all faces in video frame
                for face in faces_list:
                    found = False
                    # Check if this face has already been found
                    for face_id, target_face in main_window.target_faces.items():
                        parameters = main_window.parameters[target_face.face_id]
                        threshhold = parameters.get("SimilarityThresholdSlider", 0.6)
                        if main_window.models_processor.findCosineDistance(
                            target_face.get_embedding(
                                str(
                                    control.get(
                                        "RecognitionModelSelection", "arcface_128"
                                    )
                                )
                            ),
                            face[1],
                        ) >= float(threshhold):
                            found = True
                            break

                    if not found:
                        face_img = face[2].cpu().numpy()
                        face_img = face_img[
                            ..., ::-1
                        ]  # Swap the channels from RGB to BGR
                        face_img = numpy.ascontiguousarray(face_img)

                        # Make native Qimage
                        height, width, channel = face_img.shape
                        bytes_per_line = 3 * width
                        q_image = QtGui.QImage(
                            face_img.data,
                            width,
                            height,
                            bytes_per_line,
                            QtGui.QImage.Format_BGR888,
                        ).copy()

                        # Only store the embedding for the currently selected recognition model
                        embedding_store: Dict[str, numpy.ndarray] = {}
                        selected_recognition_model = control.get(
                            "RecognitionModelSelection", "arcface_128"
                        )

                        # The embedding for the selected model was already calculated
                        embedding_store[str(selected_recognition_model)] = face[1]

                        face_id = str(uuid.uuid1())

                        # Pass QImage instead of Pixmap
                        list_view_actions.add_media_thumbnail_to_target_faces_list(
                            main_window, face_img, embedding_store, q_image, face_id
                        )

                        new_target_face = main_window.target_faces.get(face_id)
                        if new_target_face:
                            _assign_checked_sources_to_target_face(
                                main_window, new_target_face
                            )

            # Select the first target face if no target face is already selected
            if main_window.target_faces and not main_window.selected_target_face_id:
                list(main_window.target_faces.values())[0].click()

    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()
    common_widget_actions.refresh_frame(main_window)
    video_control_actions.update_scan_review_button_states(main_window)

    common_widget_actions.update_gpu_memory_progressbar(main_window)


def _restore_target_face_scan_ui(main_window: "MainWindow") -> None:
    progress_dialog = getattr(main_window, "face_scan_progress_dialog", None)
    if progress_dialog is not None:
        progress_dialog.close()
        progress_dialog.deleteLater()
        main_window.face_scan_progress_dialog = None

    layout_actions.enable_all_parameters_and_control_widget(main_window)
    from app.ui.widgets.actions import video_control_actions

    video_control_actions.set_scan_mutation_lock_state(main_window, False)


def _cleanup_target_face_scan_worker(main_window: "MainWindow") -> None:
    worker = getattr(main_window, "face_scan_worker", None)
    if worker is not None:
        worker.deleteLater()
        main_window.face_scan_worker = None


def _handle_target_face_scan_progress(
    main_window: "MainWindow",
    processed: int,
    total: int,
    frame_number: int,
    unique_count: int,
    scan_fps: float,
) -> None:
    progress_dialog = getattr(main_window, "face_scan_progress_dialog", None)
    if progress_dialog is None:
        return
    progress_dialog.setRange(0, max(1, int(total)))
    progress_dialog.setValue(int(processed))
    progress_dialog.setLabelText(
        f"Frame {int(frame_number)} | {int(unique_count)} unique | {scan_fps:.1f} FPS"
    )


def _add_scanned_target_faces(
    main_window: "MainWindow", candidates: list[dict], scan_result: dict
) -> int:
    added_face_ids = []
    media_path = str(scan_result.get("media_path", ""))
    mode_key = str(scan_result.get("mode_key", "smart"))
    source_fps = float(scan_result.get("source_fps", 0.0))

    main_window.targetFacesList.setUpdatesEnabled(False)
    try:
        for candidate in candidates:
            cropped_face = candidate.get("cropped_face")
            embedding_store = candidate.get("embedding_store")
            if (
                not isinstance(cropped_face, numpy.ndarray)
                or cropped_face.size == 0
                or not isinstance(embedding_store, dict)
                or not embedding_store
            ):
                continue

            face_id = str(uuid.uuid1())
            list_view_actions.add_media_thumbnail_to_target_faces_list(
                main_window,
                cropped_face,
                embedding_store,
                _face_crop_to_qimage(cropped_face),
                face_id,
            )
            target_face = main_window.target_faces.get(face_id)
            if target_face is None:
                continue
            target_face.set_scan_metadata(
                frame_number=int(candidate.get("frame_number", 0)),
                media_path=media_path,
                occurrences=int(candidate.get("occurrences", 1)),
                mode_key=mode_key,
                source_fps=source_fps,
            )
            _assign_checked_sources_to_target_face(main_window, target_face)
            added_face_ids.append(face_id)
    finally:
        main_window.targetFacesList.setUpdatesEnabled(True)
        main_window.targetFacesList.viewport().update()

    if added_face_ids and not main_window.selected_target_face_id:
        main_window.target_faces[added_face_ids[0]].click()
    if added_face_ids:
        main_window.video_processor.ui_state_is_dirty = True
        common_widget_actions.refresh_frame(main_window)
        from app.ui.widgets.actions import video_control_actions

        video_control_actions.update_scan_review_button_states(main_window)
    return len(added_face_ids)


def _handle_target_face_scan_completed(
    main_window: "MainWindow", scan_result: dict
) -> None:
    _cleanup_target_face_scan_worker(main_window)
    _restore_target_face_scan_ui(main_window)

    candidates = list(scan_result.get("candidates", []))
    samples_scanned = int(scan_result.get("samples_scanned", 0))
    elapsed_seconds = float(scan_result.get("elapsed_seconds", 0.0))
    cancelled = bool(scan_result.get("cancelled", False))
    scan_fps = samples_scanned / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print(
        "[INFO] Face scan: "
        f"Mode={scan_result.get('mode_label', 'Smart')} | "
        f"Samples={samples_scanned} | Time={elapsed_seconds:.1f}s | "
        f"FPS={scan_fps:.1f} | Candidates={len(candidates)} | "
        f"Cancelled={cancelled} | Limit={scan_result.get('limit_reached', False)}"
    )

    if not candidates:
        title = "Face Scan Aborted" if cancelled else "Face Scan Complete"
        message = (
            f"Stopped after {samples_scanned} sampled frames. No new face views were found."
            if cancelled
            else f"Scanned {samples_scanned} frames in {elapsed_seconds:.1f}s. All detected views are already covered."
        )
        common_widget_actions.create_and_show_toast_message(
            main_window,
            title,
            message,
            style_type="warning" if cancelled else "success",
        )
        return

    review_dialog = FaceScanReviewDialog(
        candidates,
        partial=cancelled,
        parent=main_window,
    )
    if review_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    selected_candidates = review_dialog.selected_candidates()
    added_count = _add_scanned_target_faces(
        main_window, selected_candidates, scan_result
    )
    message = (
        f"Added {added_count} face views from {samples_scanned} sampled frames "
        f"({scan_fps:.1f} FPS)."
    )
    if scan_result.get("limit_reached", False):
        message += " The 100-result safety limit was reached."
    common_widget_actions.create_and_show_toast_message(
        main_window,
        "Face Scan Results Added",
        message,
        style_type="warning"
        if cancelled or scan_result.get("limit_reached", False)
        else "success",
    )
    common_widget_actions.update_gpu_memory_progressbar(main_window)


def _handle_target_face_scan_failed(
    main_window: "MainWindow", error_message: str
) -> None:
    _cleanup_target_face_scan_worker(main_window)
    _restore_target_face_scan_ui(main_window)
    common_widget_actions.create_and_show_messagebox(
        main_window,
        "Face Scan Failed",
        error_message,
        getattr(main_window, "scanTargetFacesButton", main_window),
    )


def start_target_face_scan(main_window: "MainWindow", mode_key: str = "smart") -> None:
    from app.ui.widgets import ui_workers
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_scan_active(main_window, "start a face scan"):
        return

    video_processor = main_window.video_processor
    if video_processor.file_type != "video" or not video_processor.media_path:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Face Scan Not Available",
            "Load a target video before scanning for unique face views.",
            main_window.scanTargetFacesButton,
        )
        return
    if not Path(video_processor.media_path).is_file():
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Face Scan Not Available",
            "The selected target video could not be found on disk.",
            main_window.scanTargetFacesButton,
        )
        return

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("[INFO] Stopped active processing before scanning unique faces.")

    try:
        worker = ui_workers.FaceScanWorker(main_window, mode_key, parent=main_window)
    except Exception as exc:
        _handle_target_face_scan_failed(main_window, str(exc))
        return

    main_window.face_scan_worker = worker
    video_control_actions.set_scan_mutation_lock_state(main_window, True)
    layout_actions.disable_all_parameters_and_control_widget(main_window)

    progress_dialog = QtWidgets.QProgressDialog(main_window)
    progress_dialog.setWindowTitle("Scanning Video Faces")
    progress_dialog.setLabelText("Preparing face scan...")
    progress_dialog.setRange(0, 0)
    progress_dialog.setCancelButtonText("Abort")
    progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setAutoClose(False)
    progress_dialog.setAutoReset(False)
    progress_dialog.canceled.connect(worker.cancel)
    main_window.face_scan_progress_dialog = progress_dialog

    worker.progress.connect(
        lambda processed,
        total,
        frame,
        unique,
        scan_fps: _handle_target_face_scan_progress(
            main_window, processed, total, frame, unique, scan_fps
        )
    )
    worker.completed.connect(
        lambda result: _handle_target_face_scan_completed(main_window, result)
    )
    worker.failed.connect(
        lambda error_message: _handle_target_face_scan_failed(
            main_window, error_message
        )
    )
    progress_dialog.show()
    worker.start()
