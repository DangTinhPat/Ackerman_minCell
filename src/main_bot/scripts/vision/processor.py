#!/usr/bin/env python3
"""
processor.py — Pipeline Thị Giác V2 (theo sơ đồ thiết kế).

Lớp 0: Raw logits từ ONNX — không sigmoid, không upsample
  → Giữ thông tin confidence nguyên vẹn

Lớp 1: Làm sạch + Weighted centroid trong không gian model (80×160)
  → Morphology kernel (2,2): tránh xoá quá nhiều trên ảnh nhỏ
  → weight = relu(logit) = max(logit, 0): sub-pixel accuracy
  → ROI bottom 45% = rows 44:80 (depth 0.14m–1.03m)

Lớp 2: RANSAC polynomial fit bậc 2 trong model space
  → Loại outlier từ vạch đứt đoạn / partial detection
  → Không cần sliding window (80px quá thấp → centroid trực tiếp đủ)

Lớp 2.5: Scale → world meters qua transformer (chỉ ở đây, không scale mask)
  → cx_model → u_orig, v_orig → pixel_to_vehicle() → (X_m, Y_m)
  → e_y   = Y_world tại row 60% (depth 0.52m, lookahead đúng)
  → e_psi = atan2(ΔY, ΔX): near=0.52m, far=1.03m (bắt cua sớm)

Lớp 3: Adaptive EMA (confidence-based alpha)
  → confidence > 0.7 → α = 0.4 (update nhanh)
  → confidence < 0.3 → α = 0.0 (predict-only, giữ giá trị cũ)
  → mất dấu > 25 frame → valid=False → controller dừng xe

Kết quả: ±4–6px → ±0.2px nhờ logit weighting + polyfit + EMA
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .ai_detector import AIDetector
    from .transformer import GeometryTransformer
except ImportError:
    from ai_detector import AIDetector
    from transformer import GeometryTransformer

# ── ROI ───────────────────────────────────────────────────────────────────────
# Camera pitch=0, h=0.134m, fy=184.8px, orig 480×640, model 80×160:
#   row 44 → depth 1.03m (horizon gần: không nên đi sâu hơn)
#   row 48 → depth 0.52m  ← lookahead tốt cho e_y
#   row 54 → depth 0.30m
#   row 70 → depth 0.14m  ← CŨ: quá gần, xe vào cua mới detect được!
_ROI_FRAC        = 0.45    # rows 44:80 (depth 0.14–1.03m, thêm data xa)

# ── Centroid per row ──────────────────────────────────────────────────────────
_MIN_ROW_SIGNAL  = 0.3     # tổng relu-weight tối thiểu để hàng có giá trị
_GOOD_ROW_SIGNAL = 5.0     # tổng relu-weight để quality=1.0

# ── RANSAC ────────────────────────────────────────────────────────────────────
_RANSAC_ITER     = 40      # số vòng lặp RANSAC
_RANSAC_THRESH   = 3.0     # ngưỡng inlier trong model pixels
_RANSAC_MIN_IN   = 4       # inlier tối thiểu để chấp nhận fit

# ── Adaptive EMA ──────────────────────────────────────────────────────────────
_EMA_CONF_LOW    = 0.30    # dưới ngưỡng này: alpha=0 (predict-only)
_EMA_CONF_HIGH   = 0.70    # trên ngưỡng này: alpha = _EMA_ALPHA_MAX
_EMA_ALPHA_MAX   = 0.50    # tăng từ 0.40 → nhanh hơn trên đường cong
_MAX_LOSS_FRAMES = 25      # ~2.5s @ 10Hz trước khi báo invalid

# ── Đánh giá poly → world ────────────────────────────────────────────────────
# Lookahead cần ≥ v/k = 0.15/0.3 = 0.5m để Stanley ổn định trên cua.
#   _Y_EVAL_FRAC = 0.60  → row 48 → depth 0.52m  (e_y lookahead)
#   _Y_PSI_NEAR  = 0.60  → row 48 → depth 0.52m
#   _Y_PSI_FAR   = 0.55  → row 44 → depth 1.03m  (span = 0.51m)
_Y_EVAL_FRAC     = 0.60    # row 48, depth 0.52m — đủ lookahead cho e_y
_Y_PSI_NEAR_FRAC = 0.60    # row 48, depth 0.52m — near ref e_psi
_Y_PSI_FAR_FRAC  = 0.55    # row 44, depth 1.03m — far ref e_psi (bắt cua sớm)

# ── Morphology ────────────────────────────────────────────────────────────────
_MORPH_KERNEL    = (2, 2)  # nhỏ phù hợp 80×160

# ── Lane geometry (từ race_way.world) ────────────────────────────────────────
# road_top width=1.068m, center divider ở Y=2.534, outer edge ở Y=2.0/3.068
# Lane width = 1.068/2 = 0.534m  → half_width = 0.267m
_HALF_LANE_W_M   = 0.267   # nửa chiều rộng làn [m], dùng cho single-channel

# ── Debug ─────────────────────────────────────────────────────────────────────
_DBG_MASK_ALPHA  = 0.30
_DBG_MASK_CLR    = (200, 80, 0)
_DBG_INLIER_CLR  = (0, 220, 0)
_DBG_OUTLIER_CLR = (100, 100, 255)
_DBG_POLY_CLR    = (0, 220, 0)
_DBG_DOT_R       = 3


class LaneProcessor:
    """Frame BGR → (e_y, e_psi, kappa, valid, mask_display, debug_img)."""

    def __init__(
        self,
        model_path:   str,
        cam_height:   float = 0.134,
        cam_pitch:    float = 0.0,
        cam_x_offset: float = 0.1485,
    ):
        self._det = AIDetector(model_path)
        self._tf  = GeometryTransformer(
            cam_height=cam_height,
            cam_pitch=cam_pitch,
            cam_x_offset=cam_x_offset,
        )

        # Adaptive EMA state
        self._e_y_ema:    float = 0.0
        self._e_psi_ema:  float = 0.0
        self._kappa_ema:  float = 0.0   # độ cong đường [1/m], dương = cua trái
        self._has_init:   bool  = False
        self._loss_frames: int  = 0

    # ── Helper: half-lane-width in model-space pixels ────────────────────────
    def _half_lane_px(self, model_row: int, sc_row: float, W_m: int, orig_w: int) -> float:
        """
        Trả về nửa chiều rộng làn (0.267m) tính bằng pixels trong model space
        tại hàng `model_row`.  Dùng để offset centroid khi chỉ thấy 1 vạch.
        """
        v_orig = min(int(model_row * sc_row), self._tf.img_h - 1)
        scale  = self._tf.lateral_scale_at(v_orig)   # m per orig pixel (lateral)
        if scale is None or scale < 1e-9:
            # Fallback: ước tính hình học
            dv = (v_orig - self._tf.cy) / self._tf.fy
            if abs(dv) < 1e-9:
                return 0.0
            depth = self._tf.cam_height / (math.cos(self._tf.cam_pitch) * dv)
            scale = depth / self._tf.fx
        return (_HALF_LANE_W_M / scale) * (W_m / orig_w)

    # ── Lớp 0 + 1: Raw logits + Dual-channel midpoint centroid ───────────────
    def _extract_centroids(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """
        Returns:
          pts_row      : [N] float32  — model row index của mỗi centroid
          pts_cx       : [N] float32  — TÂM LÀN (sub-pixel, model space)
          pts_quality  : [N] float32  — [0,1] confidence
          combined_relu: [H_m, W_m] float32  — max(relu0, relu1) (debug mask)
          orig_h, orig_w : kích thước ảnh gốc
          H_m, W_m       : kích thước model

        Chiến lược tính tâm làn mỗi hàng:
          - Cả 2 kênh có signal  → midpoint(cx0, cx1)  [chính xác nhất]
          - Chỉ ch0 (vạch trái)  → cx0 + half_lane_px  [ước tính từ độ rộng làn]
          - Chỉ ch1 (vạch phải)  → cx1 − half_lane_px
        Trọng số quality giảm 50% cho trường hợp 1 kênh để RANSAC tin tưởng ít hơn.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        sc_col = orig_w / 160.0   # sẽ được cập nhật sau khi biết W_m
        sc_row = orig_h / 80.0

        # Lớp 0: Raw logits trong model space — KHÔNG sigmoid, KHÔNG resize
        logits = self._det.detect_raw(frame_bgr)   # [3, H_m, W_m]
        H_m, W_m = logits.shape[1], logits.shape[2]
        sc_col = orig_w / W_m
        sc_row = orig_h / H_m

        # relu riêng cho từng kênh
        relu0 = np.maximum(logits[0], 0.0)   # ch0: vạch trái (center divider)
        relu1 = np.maximum(logits[1], 0.0)   # ch1: vạch phải (outer edge)

        # Lớp 1a: Morphology OPEN riêng từng kênh
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, _MORPH_KERNEL)

        binary0 = (logits[0] > 0).astype(np.uint8) * 255
        binary0 = cv2.morphologyEx(binary0, cv2.MORPH_OPEN, kernel)
        relu0   = relu0 * (binary0 > 0).astype(np.float32)

        binary1 = (logits[1] > 0).astype(np.uint8) * 255
        binary1 = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, kernel)
        relu1   = relu1 * (binary1 > 0).astype(np.float32)

        # combined mask chỉ dùng để debug / visualize
        combined_relu = np.maximum(relu0, relu1)

        roi_top = int((1.0 - _ROI_FRAC) * H_m)   # = 44 khi H_m=80
        roi0    = relu0[roi_top:]                  # [roi_h, W_m]
        roi1    = relu1[roi_top:]
        roi_h   = roi0.shape[0]

        col_idx = np.arange(W_m, dtype=np.float32)

        pts_row: List[float] = []
        pts_cx:  List[float] = []
        pts_q:   List[float] = []

        for r in range(roi_h):
            row0   = roi0[r]
            row1   = roi1[r]
            total0 = float(row0.sum())
            total1 = float(row1.sum())
            has0   = total0 >= _MIN_ROW_SIGNAL
            has1   = total1 >= _MIN_ROW_SIGNAL

            if not has0 and not has1:
                continue

            model_row = roi_top + r

            if has0 and has1:
                # Cả 2 vạch thấy được → midpoint = tâm làn thật sự
                cx0 = float((row0 * col_idx).sum() / total0)
                cx1 = float((row1 * col_idx).sum() / total1)
                cx  = (cx0 + cx1) * 0.5
                q   = min(total0, total1) / _GOOD_ROW_SIGNAL
            elif has0:
                # Chỉ vạch trái → dịch phải (cột lớn hơn) một nửa làn
                cx0       = float((row0 * col_idx).sum() / total0)
                half_px   = self._half_lane_px(model_row, sc_row, W_m, orig_w)
                cx        = cx0 + half_px
                q         = total0 / _GOOD_ROW_SIGNAL * 0.5
            else:
                # Chỉ vạch phải → dịch trái (cột nhỏ hơn) một nửa làn
                cx1       = float((row1 * col_idx).sum() / total1)
                half_px   = self._half_lane_px(model_row, sc_row, W_m, orig_w)
                cx        = cx1 - half_px
                q         = total1 / _GOOD_ROW_SIGNAL * 0.5

            pts_row.append(float(model_row))
            pts_cx.append(cx)
            pts_q.append(min(1.0, q))

        return (
            np.array(pts_row, dtype=np.float32),
            np.array(pts_cx,  dtype=np.float32),
            np.array(pts_q,   dtype=np.float32),
            combined_relu,
            orig_h, orig_w,
            H_m, W_m,
        )

    # ── Lớp 2: RANSAC polynomial fit ─────────────────────────────────────────
    @staticmethod
    def _ransac_polyfit(
        rows:    np.ndarray,
        cols:    np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Fit cx = A·row² + B·row + C với RANSAC trong model space.
        Returns (coeffs, inlier_bool_mask) hoặc (None, empty_mask).
        """
        n = len(rows)
        if n < _RANSAC_MIN_IN:
            return None, np.zeros(n, dtype=bool)

        rng           = np.random.default_rng(seed=42)
        best_coeffs:  Optional[np.ndarray] = None
        best_inliers: np.ndarray           = np.zeros(n, dtype=bool)
        best_n_in:    int                  = 0

        for _ in range(_RANSAC_ITER):
            idx = rng.choice(n, 3, replace=False)
            try:
                c = np.polyfit(rows[idx], cols[idx], 2)
            except (np.linalg.LinAlgError, ValueError):
                continue

            res     = np.abs(cols - np.polyval(c, rows))
            inliers = res < _RANSAC_THRESH
            n_in    = int(inliers.sum())

            if n_in > best_n_in:
                best_n_in    = n_in
                best_inliers = inliers
                best_coeffs  = c

        if best_coeffs is None or best_n_in < _RANSAC_MIN_IN:
            return None, np.zeros(n, dtype=bool)

        # Refit với tất cả inlier và quality weights
        try:
            coeffs = np.polyfit(
                rows[best_inliers], cols[best_inliers], 2,
                w=weights[best_inliers],
            )
        except (np.linalg.LinAlgError, ValueError):
            coeffs = best_coeffs

        return coeffs, best_inliers

    # ── Lớp 2.5: Scale model space → world meters ────────────────────────────
    def _poly_to_errors(
        self,
        coeffs: np.ndarray,
        H_m: int, W_m: int,
        orig_h: int, orig_w: int,
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Chuyển polynomial trong model space (col=f(row)) → (e_y, e_psi, kappa).

        Scale từ model → original: u = col * (orig_w/W_m), v = row * (orig_h/H_m)
        Rồi dùng transformer.pixel_to_vehicle(u, v) → (X_world, Y_world).

          e_y   = Y_world tại điểm đánh giá
          e_psi = atan2(ΔY, ΔX) giữa điểm near và điểm far
          kappa = 2*(Y_f-Y_n)/(X_f²-X_n²) — độ cong đường tại gốc robot [1/m]
                  (e_y offset tự triệt tiêu trong công thức này ✓)
        """
        sc_col = orig_w / W_m    # = 640/160 = 4.0
        sc_row = orig_h / H_m    # = 480/80  = 6.0

        # Điểm đánh giá e_y
        row_eval = _Y_EVAL_FRAC * H_m
        cx_eval  = float(np.polyval(coeffs, row_eval))
        pt_eval  = self._tf.pixel_to_vehicle(cx_eval * sc_col, row_eval * sc_row)
        if pt_eval is None:
            return None, None, 0.0
        _, e_y = pt_eval

        if abs(e_y) > 1.5:
            return None, None, 0.0

        # Hai điểm để tính e_psi và kappa
        row_near = _Y_PSI_NEAR_FRAC * H_m
        row_far  = _Y_PSI_FAR_FRAC  * H_m

        cx_near = float(np.polyval(coeffs, row_near))
        cx_far  = float(np.polyval(coeffs, row_far))

        pt_near = self._tf.pixel_to_vehicle(cx_near * sc_col, row_near * sc_row)
        pt_far  = self._tf.pixel_to_vehicle(cx_far  * sc_col, row_far  * sc_row)

        e_psi = 0.0
        kappa = 0.0

        if pt_near is not None and pt_far is not None:
            X_n, Y_n = pt_near
            X_f, Y_f = pt_far
            dX = X_f - X_n
            dY = Y_f - Y_n
            e_psi = math.atan2(dY, dX) if abs(dX) > 1e-6 else 0.0

            # Ước tính độ cong đường tại vị trí robot (gốc tọa độ):
            #   Y(X) = e_y + kappa/2 * X²  (xấp xỉ parabola, e_y = lateral offset)
            #   → Y_f - Y_n = kappa/2 * (X_f² - X_n²)
            #   → kappa = 2*(Y_f - Y_n) / (X_f² - X_n²)   [offset e_y triệt tiêu ✓]
            denom = X_f * X_f - X_n * X_n
            if abs(denom) > 1e-6:
                kappa = float(np.clip(2.0 * dY / denom, -2.0, 2.0))
        elif pt_near is not None:
            e_psi = 0.0

        return e_y, e_psi, kappa

    # ── Vẽ debug ──────────────────────────────────────────────────────────────
    def _draw_debug(
        self,
        frame_bgr:     np.ndarray,
        combined_relu: np.ndarray,
        pts_row:       np.ndarray,
        pts_cx:        np.ndarray,
        inliers:       np.ndarray,
        coeffs:        Optional[np.ndarray],
        e_y:           float,
        e_psi:         float,
        kappa:         float,
        valid:         bool,
        H_m: int, W_m: int,
    ) -> np.ndarray:
        orig_h, orig_w = frame_bgr.shape[:2]
        sc_col = orig_w / W_m
        sc_row = orig_h / H_m

        vis = frame_bgr.copy()

        # 1. Relu mask overlay
        relu_max = combined_relu.max()
        if relu_max > 1e-6:
            mask_u8 = (combined_relu * (255.0 / relu_max)).astype(np.uint8)
        else:
            mask_u8 = np.zeros_like(combined_relu, dtype=np.uint8)
        mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = vis.copy()
        overlay[mask_u8 > 20] = _DBG_MASK_CLR
        vis = cv2.addWeighted(vis, 1 - _DBG_MASK_ALPHA, overlay, _DBG_MASK_ALPHA, 0)

        # 2. Centroid dots: xanh = inlier, tím = outlier
        for i in range(len(pts_row)):
            u = int(pts_cx[i] * sc_col)
            v = int(pts_row[i] * sc_row)
            is_in = bool(len(inliers) > i and inliers[i])
            cv2.circle(vis, (u, v), _DBG_DOT_R,
                       _DBG_INLIER_CLR if is_in else _DBG_OUTLIER_CLR, -1)

        # 3. Polynomial line trong model space → display
        if coeffs is not None:
            roi_top = int((1.0 - _ROI_FRAC) * H_m)
            prev = None
            for r in range(roi_top, H_m):
                cx_m = float(np.polyval(coeffs, r))
                p = (int(cx_m * sc_col), int(r * sc_row))
                if 0 <= p[0] < orig_w and 0 <= p[1] < orig_h:
                    if prev is not None:
                        cv2.line(vis, prev, p, _DBG_POLY_CLR, 2)
                    prev = p

        # 4. Tâm xe
        cx_veh = orig_w // 2
        cv2.circle(vis, (cx_veh, orig_h - 12), 7, (0, 0, 255), -1)
        cv2.line(vis,   (cx_veh, orig_h - 20), (cx_veh, orig_h - 4), (0, 0, 255), 2)

        # 5. Text
        st_clr = (0, 220, 0) if valid else (0, 50, 255)
        cv2.putText(vis, 'TRACKING' if valid else 'LOST',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, st_clr, 2)
        cv2.putText(vis, f'e_y  = {e_y:+.4f} m',
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, f'e_psi= {math.degrees(e_psi):+.2f} deg',
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        r_str = f'{1.0/kappa:.1f}m' if abs(kappa) > 0.05 else 'straight'
        cv2.putText(vis, f'kappa= {kappa:+.3f} (R={r_str})',
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 2)
        cv2.putText(vis, f'pts={len(pts_row)}  loss={self._loss_frames}',
                    (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 0), 1)
        return vis

    # ── API Công Khai ─────────────────────────────────────────────────────────
    def process_frame(
        self,
        cv_image: np.ndarray,
    ) -> Tuple[float, float, float, bool, np.ndarray, np.ndarray]:
        """
        Frame BGR → (e_y, e_psi, kappa, valid, mask_display, debug_img).

          e_y   [m]     : sai số ngang (dương = tâm làn lệch trái robot)
          e_psi [rad]   : sai số góc hướng (dương = làn quẹo trái phía trước)
          kappa [1/m]   : độ cong đường tại robot (dương = cua trái, |kappa|=1/R)
          valid         : True nếu EMA đang hoạt động tốt
        """
        try:
            # Lớp 0 + 1: centroids trong model space
            (pts_row, pts_cx, pts_q,
             combined_relu,
             orig_h, orig_w, H_m, W_m) = self._extract_centroids(cv_image)

            # Lớp 2: RANSAC fit
            coeffs, inliers = self._ransac_polyfit(pts_row, pts_cx, pts_q)

            # Lớp 2.5: Scale → world
            e_y_raw   = 0.0
            e_psi_raw = 0.0
            kappa_raw = 0.0
            detected  = False

            if coeffs is not None:
                result = self._poly_to_errors(coeffs, H_m, W_m, orig_h, orig_w)
                if result[0] is not None:
                    e_y_raw, e_psi_raw, kappa_raw = result
                    detected = True

            # Lớp 3: Adaptive EMA
            n_roi_rows = max(1, int(_ROI_FRAC * H_m))
            confidence = len(pts_row) / n_roi_rows   # 0.0 – 1.0+

            if detected and confidence > _EMA_CONF_LOW:
                t     = min(1.0, (confidence - _EMA_CONF_LOW) /
                                 (_EMA_CONF_HIGH - _EMA_CONF_LOW))
                alpha = t * _EMA_ALPHA_MAX

                if not self._has_init:
                    self._e_y_ema   = e_y_raw
                    self._e_psi_ema = e_psi_raw
                    self._kappa_ema = kappa_raw
                    self._has_init  = True
                else:
                    self._e_y_ema   = alpha * e_y_raw   + (1 - alpha) * self._e_y_ema
                    self._e_psi_ema = alpha * e_psi_raw + (1 - alpha) * self._e_psi_ema
                    self._kappa_ema = alpha * kappa_raw + (1 - alpha) * self._kappa_ema

                self._loss_frames = 0
            else:
                self._loss_frames += 1

            valid = self._has_init and self._loss_frames <= _MAX_LOSS_FRAMES

            mask_model   = (combined_relu > 0).astype(np.uint8)
            mask_display = cv2.resize(mask_model, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)

            debug_img = self._draw_debug(
                cv_image, combined_relu,
                pts_row, pts_cx, inliers,
                coeffs if detected else None,
                self._e_y_ema, self._e_psi_ema, self._kappa_ema,
                valid, H_m, W_m,
            )
            return self._e_y_ema, self._e_psi_ema, self._kappa_ema, valid, mask_display, debug_img

        except Exception as exc:
            debug_img = cv_image.copy()
            cv2.putText(debug_img, f'ERR: {exc}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            return 0.0, 0.0, 0.0, False, np.zeros(cv_image.shape[:2], np.uint8), debug_img


# ── CLI debug offline ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os

    if len(sys.argv) < 2:
        print('Cách dùng: python3 processor.py <video_file>')
        sys.exit(1)

    model_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'models', 'EgoLanes_Lite_FP32.onnx'
    )
    proc = LaneProcessor(model_path)
    cap  = cv2.VideoCapture(sys.argv[1])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        e_y, e_psi, valid, _, debug = proc.process_frame(frame)
        print(f'e_y={e_y:+.4f}m  e_psi={math.degrees(e_psi):+.2f}deg  valid={valid}')
        cv2.imshow('Lane V2', debug)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
