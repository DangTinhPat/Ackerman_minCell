#!/usr/bin/env python3
"""
estimator.py — Khối Toán Học Đa Thức & Bộ Nhớ Làn Đường.

Nhận tập hợp điểm tọa độ mét từ tâm làn đường,
khớp đường cong đa thức bậc 2 và tính toán hai sai số điều khiển:
  e_y   — sai số lệch ngang tại điểm nhìn trước (lateral cross-track error)
  e_psi — sai số góc hướng tại mũi xe           (heading error)

KHÔNG phụ thuộc bất kỳ thư viện ROS 2 nào.

Mô hình đa thức (dạng tổng quát hỗ trợ cả đường thẳng lẫn đường cong):
──────────────────────────────────────────────────────────────────────────
  Biểu diễn trong hệ xe:  Y = A·X² + B·X + C

  Trong đó:
    X — khoảng cách tiến về phía trước từ tâm xe (m), dương = phía trước
    Y — độ lệch ngang so với tâm làn               (m), dương = lệch trái

  Tại điểm nhìn trước Ld:   e_y  = A·Ld² + B·Ld + C
  Tiếp tuyến tại mũi xe X=0: dY/dX|₀ = B → e_psi = arctan(B)
"""

import math
from typing import List, Optional, Tuple

import numpy as np

# ── Hằng số bộ ước lượng ─────────────────────────────────────────────────────
_LOOKAHEAD_M     = 1.0    # m — khoảng cách nhìn trước Ld để tính e_y
_MIN_PTS         = 8      # số điểm mét tối thiểu để fit đa thức hợp lệ
_MAX_LOSS_FRAMES = 10     # số khung hình mất dấu tối đa trước khi báo dừng


class LaneEstimator:
    """
    Ước lượng vị trí làn đường bằng khớp đa thức bậc 2 (poly-fit).

    Đặc điểm nổi bật:
    ─────────────────
    - Bộ nhớ đệm quán tính (inertia cache): Khi AI mất dấu làn đường
      (< _MIN_PTS điểm hợp lệ), xe vẫn chạy tiếp bằng hệ số đa thức
      của khung hình hợp lệ cuối cùng.
    - Tự động phát tín hiệu mất làn sau _MAX_LOSS_FRAMES khung hình liên tiếp.
    - Reset bộ nhớ khi nhận được điểm hợp lệ trở lại.
    """

    def __init__(
        self,
        lookahead_m:     float = _LOOKAHEAD_M,
        min_pts:         int   = _MIN_PTS,
        max_loss_frames: int   = _MAX_LOSS_FRAMES,
    ):
        """
        Parameters
        ----------
        lookahead_m     : Khoảng cách nhìn trước Ld để đánh giá e_y (m).
        min_pts         : Số điểm tối thiểu để chấp nhận kết quả fit.
        max_loss_frames : Số khung hình mất dấu tối đa trước khi báo không hợp lệ.
        """
        self._ld             = lookahead_m
        self._min_pts        = min_pts
        self._max_loss_frames = max_loss_frames

        # Bộ nhớ đệm quán tính: hệ số đa thức [A, B, C] của lần fit thành công gần nhất
        self._cached_coeffs: Optional[np.ndarray] = None  # None = chưa có dữ liệu
        self._loss_frames: int = 0                         # đếm số khung hình mất dấu

    # ──────────────────────────────────────────────────────────────────────────
    # Nội bộ: khớp đa thức bậc 2 từ danh sách điểm mét
    # ──────────────────────────────────────────────────────────────────────────
    def _polyfit(
        self, pts: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Khớp đa thức bậc 2: Y = A·X² + B·X + C lên tập hợp điểm mét.

        Sử dụng phương pháp bình phương tối thiểu (Least Squares) qua np.polyfit.

        np.polyfit(Xv, Yv, 2) trả về [A, B, C] theo thứ tự bậc giảm dần:
          Yv ≈ A · Xv² + B · Xv + C

        Returns
        -------
        np.ndarray([A, B, C]) hoặc None nếu fit thất bại.
        """
        arr = np.array(pts, dtype=np.float64)
        Xv  = arr[:, 0]   # tọa độ tiến (forward, m)
        Yv  = arr[:, 1]   # tọa độ ngang (lateral, m)

        try:
            coeffs = np.polyfit(Xv, Yv, deg=2)  # → [A, B, C]
        except (np.linalg.LinAlgError, ValueError):
            return None

        return coeffs

    # ──────────────────────────────────────────────────────────────────────────
    # Nội bộ: tính sai số từ hệ số đa thức đã khớp
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_errors(
        self, coeffs: np.ndarray
    ) -> Tuple[float, float]:
        """
        Tính toán cặp sai số điều khiển từ hệ số đa thức [A, B, C].

        Mô hình đa thức: Y = A·X² + B·X + C

        Sai số lệch ngang e_y tại điểm nhìn trước Ld:
        ─────────────────────────────────────────────
          e_y = A·Ld² + B·Ld + C
          → Đây là khoảng cách ngang giữa tâm xe và tâm làn đường
            tại khoảng cách nhìn trước, dùng để điều chỉnh tay lái.

        Sai số góc hướng e_psi tại mũi xe (X = 0):
        ─────────────────────────────────────────────
          dY/dX|_{X=0} = 2·A·0 + B = B
          → Tiếp tuyến của làn đường tại vị trí hiện tại của xe.
          e_psi = arctan(B)
          → Góc giữa hướng xe và hướng tiếp tuyến của làn đường.
        """
        A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        # Sai số lệch ngang tại điểm nhìn trước Ld (mét)
        e_y = A * self._ld**2 + B * self._ld + C

        # Sai số góc hướng: arctan của độ dốc tiếp tuyến tại X=0 (rad)
        e_psi = math.atan(B)

        return e_y, e_psi

    # ──────────────────────────────────────────────────────────────────────────
    # API công khai: ước lượng sai số từ danh sách điểm mét
    # ──────────────────────────────────────────────────────────────────────────
    def estimate(
        self, pts_meters: List[Tuple[float, float]]
    ) -> Tuple[float, float, bool]:
        """
        Ước lượng sai số điều khiển từ tập điểm tâm làn đường (đơn vị mét).

        Cơ chế bộ nhớ đệm quán tính (Inertia Cache):
        ──────────────────────────────────────────────
          - Nếu đủ điểm: fit đa thức mới, cập nhật cache, reset loss_frames.
          - Nếu thiếu điểm (mất dấu làn):
              * Dùng cache nếu còn hợp lệ (loss_frames ≤ max_loss_frames).
              * Báo invalid (valid=False) và trả về (0, 0) nếu cache hết hạn.
            → Cơ chế này chống chớp nháy do AI bỏ lỡ vài khung hình.

        Parameters
        ----------
        pts_meters : Danh sách (X_m, Y_m) — tọa độ tâm làn đường, đơn vị mét.

        Returns
        -------
        e_y   : float  Sai số lệch ngang tại điểm nhìn trước (m).
        e_psi : float  Sai số góc hướng tại mũi xe (rad).
        valid : bool   True nếu sai số từ dữ liệu hợp lệ (mới hoặc từ cache).
        """
        # ── Trường hợp 1: Đủ điểm để fit đa thức mới ────────────────────
        if len(pts_meters) >= self._min_pts:
            coeffs = self._polyfit(pts_meters)
            if coeffs is not None:
                # Fit thành công → cập nhật cache và reset bộ đếm mất dấu
                self._cached_coeffs = coeffs
                self._loss_frames   = 0
                e_y, e_psi = self._compute_errors(coeffs)
                return e_y, e_psi, True

        # ── Trường hợp 2: Thiếu điểm (mất dấu làn đường) ────────────────
        self._loss_frames += 1

        if self._cached_coeffs is not None and self._loss_frames <= self._max_loss_frames:
            # Dùng hệ số đa thức từ khung hình hợp lệ cuối cùng (quán tính)
            e_y, e_psi = self._compute_errors(self._cached_coeffs)
            return e_y, e_psi, True

        # ── Trường hợp 3: Cache đã hết hạn hoặc chưa có dữ liệu ──────────
        # Báo mất làn và trả về sai số bằng 0 để node ROS xử lý dừng khẩn cấp
        return 0.0, 0.0, False

    def reset(self):
        """Xoá toàn bộ bộ nhớ cache — dùng khi khởi động lại hoặc chuyển địa hình."""
        self._cached_coeffs = None
        self._loss_frames   = 0

    @property
    def loss_frames(self) -> int:
        """Số khung hình mất dấu liên tiếp tính đến hiện tại."""
        return self._loss_frames

    @property
    def has_cache(self) -> bool:
        """True nếu bộ nhớ cache đang chứa hệ số đa thức hợp lệ."""
        return self._cached_coeffs is not None
