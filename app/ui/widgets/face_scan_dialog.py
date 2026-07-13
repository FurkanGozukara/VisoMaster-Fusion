from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds_remainder = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:04.1f}"
    return f"{minutes:02d}:{seconds_remainder:04.1f}"


def _face_crop_to_icon(cropped_face, icon_size: QtCore.QSize) -> QtGui.QIcon:
    height, width, _channels = cropped_face.shape
    image = QtGui.QImage(
        cropped_face.data,
        width,
        height,
        int(cropped_face.strides[0]),
        QtGui.QImage.Format.Format_BGR888,
    ).copy()
    pixmap = QtGui.QPixmap.fromImage(image).scaled(
        icon_size,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    return QtGui.QIcon(pixmap)


class FaceScanReviewDialog(QtWidgets.QDialog):
    def __init__(self, candidates: list[dict], partial: bool = False, parent=None):
        super().__init__(parent)
        self._candidates = candidates
        self.setWindowTitle(
            "Review Partial Face Scan" if partial else "Review Scanned Faces"
        )
        self.setModal(True)
        self.resize(780, 560)
        self.setMinimumSize(620, 420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        result_label = QtWidgets.QLabel(f"{len(candidates)} unique face views", self)
        result_font = result_label.font()
        result_font.setBold(True)
        result_label.setFont(result_font)
        layout.addWidget(result_label)

        self.resultsList = QtWidgets.QListWidget(self)
        self.resultsList.setObjectName("faceScanResultsList")
        self.resultsList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.resultsList.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.resultsList.setMovement(QtWidgets.QListView.Movement.Static)
        self.resultsList.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.resultsList.setSpacing(6)
        self.resultsList.setIconSize(QtCore.QSize(104, 104))
        self.resultsList.setGridSize(QtCore.QSize(132, 146))
        self.resultsList.setUniformItemSizes(True)

        for index, candidate in enumerate(candidates):
            occurrences = int(candidate.get("occurrences", 1))
            timestamp = _format_timestamp(candidate.get("timestamp_seconds", 0.0))
            item = QtWidgets.QListWidgetItem(
                _face_crop_to_icon(
                    candidate["cropped_face"], self.resultsList.iconSize()
                ),
                f"{timestamp}\n{occurrences} hit{'s' if occurrences != 1 else ''}",
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            item.setToolTip(
                f"Frame {int(candidate.get('frame_number', 0))} | "
                f"{timestamp} | {occurrences} sampled detections"
            )
            self.resultsList.addItem(item)
        layout.addWidget(self.resultsList, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        select_all_button = QtWidgets.QPushButton("Select All", self)
        select_none_button = QtWidgets.QPushButton("Select None", self)
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.addButton = QtWidgets.QPushButton("Add Selected", self)
        self.addButton.setDefault(True)

        select_all_button.clicked.connect(
            lambda: self._set_all_check_states(QtCore.Qt.CheckState.Checked)
        )
        select_none_button.clicked.connect(
            lambda: self._set_all_check_states(QtCore.Qt.CheckState.Unchecked)
        )
        cancel_button.clicked.connect(self.reject)
        self.addButton.clicked.connect(self.accept)
        self.resultsList.itemChanged.connect(self._update_add_button)

        footer.addWidget(select_all_button)
        footer.addWidget(select_none_button)
        footer.addStretch(1)
        footer.addWidget(cancel_button)
        footer.addWidget(self.addButton)
        layout.addLayout(footer)
        self._update_add_button()

    def _set_all_check_states(self, state: QtCore.Qt.CheckState) -> None:
        self.resultsList.blockSignals(True)
        for item_index in range(self.resultsList.count()):
            self.resultsList.item(item_index).setCheckState(state)
        self.resultsList.blockSignals(False)
        self._update_add_button()

    def _update_add_button(self, _item=None) -> None:
        checked_count = sum(
            self.resultsList.item(item_index).checkState()
            == QtCore.Qt.CheckState.Checked
            for item_index in range(self.resultsList.count())
        )
        self.addButton.setText(f"Add Selected ({checked_count})")
        self.addButton.setEnabled(checked_count > 0)

    def selected_candidates(self) -> list[dict]:
        selected = []
        for item_index in range(self.resultsList.count()):
            item = self.resultsList.item(item_index)
            if item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            candidate_index = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
            selected.append(self._candidates[candidate_index])
        return selected
