"""DeepFaceLab / DeepFaceLive compatible face alignment.

DFM models are DeepFaceLab models exported to ONNX.  They are trained on faces
extracted with DFL's ``LandmarksProcessor.get_transform_mat()``, which frames the
face far wider than the ArcFace template VisoMaster uses for its own swappers
(Inswapper/SimSwap/InStyle).  Feeding a DFM model an ArcFace-framed crop puts it
outside its training distribution: the face is cropped too tightly, proportions
are off, and identity/detail transfer degrades noticeably.

This module reproduces DFL's alignment maths so a DFM model receives exactly the
framing it was trained on.

Reference geometry
------------------
DFL builds the alignment in two steps:

1. Fit a similarity transform (umeyama) from the detected landmarks onto a
   canonical "uni" face template that lives in the unit square ``[0..1]^2``.
2. Take the unit square's corners back into image space and expand them
   outwards by a *coverage* factor around the face centre.  The expanded square
   becomes the crop.

DFL expresses step 2 as a per-face-type ``padding`` value::

    mod = diag * (padding * sqrt(2) + 0.5)

DeepFaceLive expresses the same thing as a ``coverage`` slider::

    mod = diag * (coverage * 0.5)

so ``coverage == 2 * (padding * sqrt(2) + 0.5)``.  Both conventions are
supported here; :data:`FACE_TYPE_COVERAGE` holds the exact equivalents.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# DFL canonical template (``landmarks_2D_new`` in DeepFaceLab).
# 33 points: 68-point indices 17..48 followed by index 54.
# ---------------------------------------------------------------------------
_LANDMARK_INDICES: tuple[int, ...] = tuple(range(17, 49)) + (54,)

LANDMARKS_2D_NEW = np.array(
    [
        [0.000213256, 0.106454],  # 17
        [0.0752622, 0.038915],  # 18
        [0.18113, 0.0187482],  # 19
        [0.29077, 0.0344891],  # 20
        [0.393397, 0.0773906],  # 21
        [0.586856, 0.0773906],  # 22
        [0.689483, 0.0344891],  # 23
        [0.799124, 0.0187482],  # 24
        [0.904991, 0.038915],  # 25
        [0.98004, 0.106454],  # 26
        [0.490127, 0.203352],  # 27
        [0.490127, 0.307009],  # 28
        [0.490127, 0.409805],  # 29
        [0.490127, 0.515625],  # 30
        [0.36688, 0.587326],  # 31
        [0.426036, 0.609345],  # 32
        [0.490127, 0.628106],  # 33
        [0.554217, 0.609345],  # 34
        [0.613373, 0.587326],  # 35
        [0.121737, 0.216423],  # 36
        [0.187122, 0.178758],  # 37
        [0.265825, 0.179852],  # 38
        [0.334606, 0.231733],  # 39
        [0.260918, 0.245099],  # 40
        [0.182743, 0.244077],  # 41
        [0.645647, 0.231733],  # 42
        [0.714428, 0.179852],  # 43
        [0.793132, 0.178758],  # 44
        [0.858516, 0.216423],  # 45
        [0.79751, 0.244077],  # 46
        [0.719335, 0.245099],  # 47
        [0.254149, 0.780233],  # 48
        [0.726104, 0.780233],  # 54
    ],
    dtype=np.float64,
)

_TEMPLATE_POS = {idx: i for i, idx in enumerate(_LANDMARK_INDICES)}


def _template_point(idx: int) -> np.ndarray:
    return LANDMARKS_2D_NEW[_TEMPLATE_POS[idx]]


def _template_mean(indices: range) -> np.ndarray:
    return LANDMARKS_2D_NEW[[_TEMPLATE_POS[i] for i in indices]].mean(axis=0)


# The same five semantic points ArcFace uses (left eye, right eye, nose tip,
# left mouth corner, right mouth corner), expressed in DFL "uni" coordinates.
# Derived from the template above so the two stay in sync.
UNI_LANDMARKS_5 = np.stack(
    [
        _template_mean(range(36, 42)),  # subject's right eye (viewer left)
        _template_mean(range(42, 48)),  # subject's left eye  (viewer right)
        _template_point(30),  # nose tip
        _template_point(48),  # mouth corner, viewer left
        _template_point(54),  # mouth corner, viewer right
    ]
).astype(np.float64)


def _padding_to_coverage(padding: float) -> float:
    return 2.0 * (padding * np.sqrt(2.0) + 0.5)


#: DFL ``FaceType`` -> DeepFaceLive ``coverage`` equivalents.
FACE_TYPE_COVERAGE: dict[str, float] = {
    "half": _padding_to_coverage(0.0),  # 1.0000
    "mid_full": _padding_to_coverage(0.0675),  # 1.1909
    "full": _padding_to_coverage(0.2109375),  # 1.5966
    "whole_face": _padding_to_coverage(0.40),  # 2.1314
    "head": _padding_to_coverage(0.70),  # 2.9799
}

#: Vertical nudge (in units of the uni-square side) DFL applies for WHOLE_FACE
#: so the crop covers more forehead.  Negative moves the crop upwards.
WHOLE_FACE_Y_OFFSET: float = -0.07

#: Default framing for DFM models (DFL ``whole_face``, the face type virtually
#: every published DFM model is trained with).
DEFAULT_COVERAGE: float = FACE_TYPE_COVERAGE["whole_face"]


# ---------------------------------------------------------------------------
# Small affine helpers (2x3 row-major matrices, the OpenCV/kornia convention)
# ---------------------------------------------------------------------------
def to_3x3(mat: np.ndarray) -> np.ndarray:
    """Promotes a 2x3 affine matrix to its 3x3 homogeneous form."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (3, 3):
        return mat
    out: np.ndarray = np.eye(3, dtype=np.float64)
    out[:2] = mat
    return out


def invert_affine(mat: np.ndarray) -> np.ndarray:
    """Inverts a 2x3 affine transform, returning a 2x3 matrix."""
    return np.linalg.inv(to_3x3(mat))[:2]


def compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Returns the 2x3 affine equivalent to applying *inner* then *outer*."""
    return (to_3x3(outer) @ to_3x3(inner))[:2]


def transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Applies a 2x3 affine transform to an ``(N, 2)`` array of points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mat = np.asarray(mat, dtype=np.float64)[:2]
    return pts @ mat[:, :2].T + mat[:, 2]


def _affine_from_3_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solves the exact 2x3 affine mapping three source points onto three
    destination points (the numpy equivalent of ``cv2.getAffineTransform``)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    a = np.hstack([src, np.ones((3, 1), dtype=np.float64)])
    # Solve a @ x = dst for each output axis; `solve` raises on singular input.
    return np.linalg.solve(a, dst).T


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation + uniform scale +
    translation) mapping *src* onto *dst*.  Returns a 2x3 matrix.

    Implemented locally (rather than importing ``faceutil.umeyama``) to keep this
    module free of heavy imports and to work in float64 throughout.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var = src_demean.var(axis=0).sum()
    if not np.isfinite(src_var) or src_var < 1e-12:
        raise ValueError("degenerate landmark set (zero variance)")

    a = (dst_demean.T @ src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1.0

    u, s, vt = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        raise ValueError("degenerate landmark set (rank 0)")

    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rot = u @ vt
        else:
            s_last = d[dim - 1]
            d[dim - 1] = -1.0
            rot = u @ np.diag(d) @ vt
            d[dim - 1] = s_last
    else:
        rot = u @ np.diag(d) @ vt

    scale = float(s @ d) / src_var
    mat: np.ndarray = np.empty((2, 3), dtype=np.float64)
    mat[:, :2] = rot * scale
    mat[:, 2] = dst_mean - scale * (rot @ src_mean)
    if not np.all(np.isfinite(mat)):
        raise ValueError("similarity transform produced NaN/Inf")
    return mat


# ---------------------------------------------------------------------------
# image -> uni-space transforms
# ---------------------------------------------------------------------------
def uni_mat_from_landmarks_68(landmarks_68: np.ndarray) -> np.ndarray:
    """Exact DFL fit: maps image coordinates onto the uni square using the same
    33 landmarks (17..48 plus 54) DeepFaceLab uses.

    Args:
        landmarks_68: ``(68, 2)`` or ``(68, 3)`` ibug-68 landmarks in image space.

    Returns:
        A 2x3 affine mapping image coordinates to uni ``[0..1]`` coordinates.
    """
    lmk = np.asarray(landmarks_68, dtype=np.float64)
    if lmk.ndim != 2 or lmk.shape[0] != 68:
        raise ValueError(f"expected 68 landmarks, got shape {lmk.shape}")
    lmk = lmk[:, :2]
    src = np.concatenate([lmk[17:49], lmk[54:55]], axis=0)
    return _umeyama_similarity(src, LANDMARKS_2D_NEW)


def uni_mat_from_kps_5(kps_5: np.ndarray) -> np.ndarray:
    """Approximate DFL fit from the 5-point keypoints VisoMaster already has.

    The five ArcFace points (eye centres, nose tip, mouth corners) have exact
    counterparts in the DFL template, so a similarity fit between them recovers
    the same framing as the 33-point fit to within a few percent of face size —
    without requiring a 68-point landmark model, and without adding any
    per-frame jitter of its own.

    Args:
        kps_5: ``(5, 2)`` keypoints ordered
            ``[eye_left, eye_right, nose, mouth_left, mouth_right]``.

    Returns:
        A 2x3 affine mapping image coordinates to uni ``[0..1]`` coordinates.
    """
    kps = np.asarray(kps_5, dtype=np.float64).reshape(-1, 2)
    if kps.shape[0] != 5:
        raise ValueError(f"expected 5 keypoints, got shape {kps.shape}")
    return _umeyama_similarity(kps, UNI_LANDMARKS_5)


# ---------------------------------------------------------------------------
# uni-space transform -> final crop matrix
# ---------------------------------------------------------------------------
def transform_mat_from_uni(
    uni_mat: np.ndarray,
    output_size: int,
    coverage: float = DEFAULT_COVERAGE,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    """Expands the uni square by *coverage* and returns the final crop matrix.

    This is DeepFaceLab's ``get_transform_mat`` corner maths, parameterised with
    DeepFaceLive's ``coverage`` instead of DFL's per-face-type ``padding``.

    Args:
        uni_mat:     2x3 affine mapping image -> uni space.
        output_size: Side length in pixels of the square crop to produce.
        coverage:    Framing factor; see :data:`FACE_TYPE_COVERAGE`.
        x_offset:    Horizontal nudge in units of the uni-square side
                     (positive moves the crop right).
        y_offset:    Vertical nudge in units of the uni-square side
                     (negative moves the crop up, revealing more forehead).

    Returns:
        A 2x3 affine mapping image coordinates to the ``output_size`` crop.
    """
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    inv_uni = invert_affine(uni_mat)

    # Unit-square corners (+ centre) taken back into image space.
    g_p = transform_points(
        np.array(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)],
            dtype=np.float64,
        ),
        inv_uni,
    )
    g_c = g_p[4].copy()

    # Diagonal unit vectors: top-left -> bottom-right and bottom-left -> top-right.
    tb_diag_vec = g_p[2] - g_p[0]
    bt_diag_vec = g_p[1] - g_p[3]
    tb_norm = np.linalg.norm(tb_diag_vec)
    bt_norm = np.linalg.norm(bt_diag_vec)
    if tb_norm < 1e-9 or bt_norm < 1e-9:
        raise ValueError("degenerate uni transform")
    tb_diag_vec = tb_diag_vec / tb_norm
    bt_diag_vec = bt_diag_vec / bt_norm

    # Half-diagonal of the final crop, in image pixels.
    mod = float(np.linalg.norm(g_p[0] - g_p[2]) * (coverage * 0.5))

    if x_offset or y_offset:
        h_vec = g_p[1] - g_p[0]  # uni +X in image space
        v_vec = g_p[3] - g_p[0]  # uni +Y in image space
        g_c = g_c + h_vec * float(x_offset) + v_vec * float(y_offset)

    src_tri = np.stack(
        [
            g_c - tb_diag_vec * mod,  # top-left
            g_c + bt_diag_vec * mod,  # top-right
            g_c + tb_diag_vec * mod,  # bottom-right
        ]
    )
    dst_tri = np.array(
        [
            (0.0, 0.0),
            (float(output_size), 0.0),
            (float(output_size), float(output_size)),
        ],
        dtype=np.float64,
    )
    return _affine_from_3_pairs(src_tri, dst_tri)


def get_dfl_transform(
    output_size: int,
    kps_5: np.ndarray | None = None,
    landmarks_68: np.ndarray | None = None,
    coverage: float = DEFAULT_COVERAGE,
    x_offset: float = 0.0,
    y_offset: float = WHOLE_FACE_Y_OFFSET,
) -> np.ndarray:
    """Builds the image -> DFL crop matrix, preferring 68-point landmarks.

    Args:
        output_size:  Side length of the square crop (usually the DFM model's
                      native input resolution).
        kps_5:        ``(5, 2)`` ArcFace-ordered keypoints; used when
                      *landmarks_68* is unavailable or unusable.
        landmarks_68: ``(68, 2|3)`` ibug-68 landmarks for the exact DFL fit.
        coverage:     Framing factor; see :data:`FACE_TYPE_COVERAGE`.
        x_offset:     Horizontal nudge in uni-square-side units.
        y_offset:     Vertical nudge in uni-square-side units.

    Returns:
        A 2x3 affine mapping image coordinates to the crop.

    Raises:
        ValueError: If neither landmark set is usable.
    """
    uni_mat = None
    if landmarks_68 is not None:
        lmk = np.asarray(landmarks_68)
        if lmk.ndim == 2 and lmk.shape[0] == 68 and np.all(np.isfinite(lmk[:, :2])):
            try:
                uni_mat = uni_mat_from_landmarks_68(lmk)
            except ValueError:
                uni_mat = None

    if uni_mat is None:
        if kps_5 is None:
            raise ValueError("no usable landmarks supplied")
        uni_mat = uni_mat_from_kps_5(kps_5)

    return transform_mat_from_uni(
        uni_mat,
        output_size,
        coverage=coverage,
        x_offset=x_offset,
        y_offset=y_offset,
    )


def coverage_of_crop(
    crop_mat: np.ndarray, uni_mat: np.ndarray, output_size: int
) -> float:
    """Returns the DFL ``coverage`` a given crop matrix corresponds to.

    Useful for comparing a non-DFL crop (e.g. VisoMaster's ArcFace 512 crop)
    against DFL face types.  The uni square has unit sides, so the crop's side
    length measured in uni units *is* its coverage.
    """
    crop_to_uni = compose_affine(uni_mat, invert_affine(crop_mat))
    corners = transform_points(
        np.array([(0.0, 0.0), (float(output_size), 0.0)], dtype=np.float64), crop_to_uni
    )
    return float(np.linalg.norm(corners[1] - corners[0]))
