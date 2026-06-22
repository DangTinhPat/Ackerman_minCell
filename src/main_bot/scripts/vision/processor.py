#!/usr/bin/env python3
"""
processor.py — Bộ Điều Phối Pipeline Thị Giác Chính (Vision Pipeline Orchestrator).

Khởi tạo và kết nối ba khối chức năng:
  AIDetector        → phát hiện vạch làn (AI Inference)
  GeometryTransformer → biến đổi hình học pixel ↔ mét
  LaneEstimator     → khớp đa thức và tính sai số điều khiển

Giao diện công khai duy nhất: process_frame(cv_image) → (e_y, e_psi, mask, debug_img)

KHÔNG phụ thuộc bất kỳ thư viện ROS 2 nào — có thể test offline bằng video.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import cv2

try:
    from .ai_detector  import AIDetector
    from .transformer  import GeometryTransformer
    from .estimator    import LaneEstimator
except ImportError:
    from ai_detector  import AIDetector
    from transformer  import GeometryTransformer
    from estimator    import LaneEstimator

# ── Tham số trích xuất tâm làn đường ──────────────────────────────────────────
_MIN_LANE_GAP_PX       = 30     # px — khoảng cách pixel tối thiểu giữa 2 vạch để
                                 #       xác định "đây là 2 vạch riêng biệt"
_ASSUMED_HALF_WIDTH_M  = 0.22   # m — nửa chiều rộng làn giả định khi chỉ thấy 1 vạch
_MIN_CLUSTER_SIZE      = 3      # pixel — số pixel tối thiểu để coi là một vạch thật

# ── Tham số vẽ debug ──────────────────────────────────────────────────────────
_DEBUG_LINE_COLOR    = (0, 220, 0)   # xanh lá — đường tâm làn dự báo
_DEBUG_DOT_RADIUS    = 4             # px — bán kính mỗi chấm trên nét chấm
_DEBUG_DOT_STEP      = 0.12          # m — khoảng cách giữa các chấm (theo X mét)
_DEBUG_MASK_COLOR_L  = (255, 100,  0)  # xanh dương — vùng vạch làn
_DEBUG_MASK_ALPHA    = 0.40          # độ trong suốt overlay mặt nạ


class LaneProcessor:
    """
    Bộ điều phối chính: nhận ảnh BGR thô → trả về sai số điều khiển + ảnh debug.

    Luồng dữ liệu (Data Flow):
    ──────────────────────────
    BGR frame
        │
        ▼ AIDetector.detect()
    Binary mask (640×480)
        │
        ▼ _extract_center_pts()  [nội bộ processor]
    Danh sách điểm mét (X, Y)
        │
        ▼ LaneEstimator.estimate()
    (e_y, e_psi, valid)
        │
        ▼ _draw_debug()
    Debug image
        │
        ▼ process_frame() returns
    (e_y, e_psi, mask, debug_img)
    """

    def __init__(
        self,
        model_path:  str,
        cam_height:  float = 0.134,
        cam_pitch:   float = 0.0,
        cam_x_offset: float = 0.1485,
    ):
        """
        Parameters
        ----------
        model_path   : Đường dẫn tới file .onnx (EgoLanes_Lite_FP32.onnx).
        cam_height   : Chiều cao camera so với mặt đất (m).
        cam_pitch    : Góc nghiêng camera — dương = cúi xuống (rad).
        cam_x_offset : Khoảng cách camera từ tâm xe theo hướng X (m).
        """
        # ── Khởi tạo 3 khối chức năng con ─────────────────────────────────
        self._detector   = AIDetector(model_path)
        self._transformer = GeometryTransformer(
            cam_height=cam_height,
            cam_pitch=cam_pitch,
            cam_x_offset=cam_x_offset,
        )
        self._estimator  = LaneEstimator()

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC TRUNG GIAN: Trích xuất điểm tâm làn từ mặt nạ nhị phân
    # ──────────────────────────────────────────────────────────────────────────
    def _extract_center_pts(
        self, mask: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Quét từng hàng ảnh để tìm tâm làn đường và chuyển sang toạ độ mét.

        Thuật toán theo hàng (row-by-row):
        ───────────────────────────────────
        1. Tìm tất cả pixel trắng trong hàng.
        2. Phát hiện 2 cụm (cluster) bằng cách tìm khoảng cách lớn giữa các pixel:
           - Cụm bên trái  = vạch làn TRÁI  (pixel có cột nhỏ nhất)
           - Cụm bên phải = vạch làn PHẢI (pixel có cột lớn nhất)
        3a. Cả 2 vạch → tâm = trung bình giữa mean_trái và mean_phải.
        3b. Chỉ 1 vạch → ước tính tâm từ chiều rộng làn giả định:
              - Nếu vạch ở nửa trái ảnh (u < cx) → đây là vạch trái
                tâm ≈ u_vạch_trái + nửa_chiều_rộng_làn_pixel
              - Nếu vạch ở nửa phải ảnh (u ≥ cx) → đây là vạch phải
                tâm ≈ u_vạch_phải − nửa_chiều_rộng_làn_pixel
        4. Gọi GeometryTransformer.pixel_to_vehicle() để đổi (u, v) → (X, Y) mét.
        """
        H, W = mask.shape
        cx = W / 2.0
        pts_meters: List[Tuple[float, float]] = []

        for v in range(H // 2, H):  # Chỉ xét nửa dưới ảnh (phần đường)
            cols = np.where(mask[v] > 0)[0]
            if len(cols) < _MIN_CLUSTER_SIZE:
                continue  # Hàng không đủ pixel

            # ── Tìm điểm ngắt cluster: khoảng cách giữa các pixel liền kề ──
            gaps    = np.diff(cols.astype(np.int32))
            split_i = np.where(gaps >= _MIN_LANE_GAP_PX)[0]

            u_center: Optional[float] = None

            if len(split_i) > 0:
                # Có ít nhất 1 khoảng trống lớn → 2+ cụm riêng biệt
                # Lấy cụm đầu tiên (trái nhất) và cụm cuối cùng (phải nhất)
                right_start = split_i[-1] + 1  # chỉ số bắt đầu cụm phải nhất
                u_left  = float(cols[:split_i[0] + 1].mean())   # mean cụm trái
                u_right = float(cols[right_start:].mean())       # mean cụm phải
                u_center = (u_left + u_right) / 2.0

            else:
                # Chỉ có một cụm liên tục → chỉ thấy 1 vạch làn
                u_mean = float(cols.mean())
                scale  = self._transformer.lateral_scale_at(v)  # m/pixel tại hàng v
                if scale is None:
                    continue

                half_px = _ASSUMED_HALF_WIDTH_M / scale  # nửa chiều rộng làn (pixel)

                if u_mean < cx:
                    # Đây là vạch làn TRÁI → tâm làn nằm bên phải của vạch này
                    u_center = u_mean + half_px
                else:
                    # Đây là vạch làn PHẢI → tâm làn nằm bên trái của vạch này
                    u_center = u_mean - half_px

            # ── Chuyển pixel (u_center, v) → mét (X, Y) ───────────────────
            if u_center is not None:
                pt = self._transformer.pixel_to_vehicle(u_center, float(v))
                if pt is not None:
                    pts_meters.append(pt)

        return pts_meters

    # ──────────────────────────────────────────────────────────────────────────
    # Vẽ ảnh debug: overlay mặt nạ + đường tâm làn nét chấm xanh
    # ──────────────────────────────────────────────────────────────────────────
    def _draw_debug(
        self,
        frame_bgr: np.ndarray,
        mask:      np.ndarray,
        e_y:       float,
        e_psi:     float,
        valid:     bool,
        coeffs:    Optional[np.ndarray],   # [A, B, C] nếu có; None nếu không
    ) -> np.ndarray:
        """
        Vẽ thông tin debug lên ảnh:
          - Overlay bán trong suốt màu xanh dương cho vùng vạch làn.
          - Đường tâm làn dự báo bằng nét chấm màu xanh lá (nếu valid).
          - Văn bản trạng thái, e_y, e_psi ở góc trái.
        """
        vis = frame_bgr.copy()

        # ── 1. Overlay mặt nạ làn đường ───────────────────────────────────
        overlay = vis.copy()
        overlay[mask > 0] = _DEBUG_MASK_COLOR_L
        vis = cv2.addWeighted(vis, 1.0 - _DEBUG_MASK_ALPHA,
                              overlay, _DEBUG_MASK_ALPHA, 0)

        # ── 2. Đường tâm làn dự báo bằng nét chấm xanh lá ────────────────
        # Chiếu chuỗi điểm theo đường cong Y = A·X² + B·X + C
        # rồi vẽ các chấm tròn cách đều nhau _DEBUG_DOT_STEP mét.
        if valid and coeffs is not None:
            A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            x_vals = np.arange(0.05, 3.5, _DEBUG_DOT_STEP)  # từ gần → xa
            for x_m in x_vals:
                y_m = A * x_m**2 + B * x_m + C           # Y (mét) theo đa thức
                px  = self._transformer.vehicle_to_pixel(x_m, y_m)
                if px is not None:
                    cv2.circle(vis, px, _DEBUG_DOT_RADIUS,
                               _DEBUG_LINE_COLOR, -1)  # chấm đặc

        # ── 3. Chấm đỏ: vị trí tâm xe ở đáy ảnh ──────────────────────────
        H, W = vis.shape[:2]
        vc   = (W // 2, H - 15)
        cv2.circle(vis, vc, 8, (0, 0, 255), -1)
        cv2.line(vis, (vc[0], vc[1] - 14), (vc[0], vc[1] + 14), (0, 0, 255), 2)

        # ── 4. Văn bản trạng thái ──────────────────────────────────────────
        status_txt   = 'TRACKING' if valid else 'NO LANE'
        status_color = (0, 220, 0) if valid else (0, 50, 255)

        cv2.putText(vis, status_txt,
                    (10, 32),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(vis, f'e_y  = {e_y:+.4f} m',
                    (10, 62),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(vis, f'e_psi= {math.degrees(e_psi):+.2f} deg',
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        if not valid and self._estimator.has_cache:
            cv2.putText(vis, f'Cache [{self._estimator.loss_frames}/{LaneEstimator.__init__.__defaults__}]',
                        (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)

        return vis

    # ──────────────────────────────────────────────────────────────────────────
    # API CÔNG KHAI: Xử lý một khung hình camera
    # ──────────────────────────────────────────────────────────────────────────
    def process_frame(
        self, cv_image: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Pipeline xử lý đầy đủ cho một khung hình camera.

        Luồng dữ liệu:
          cv_image  → [AIDetector]       → mask (640×480 binary)
          mask      → [_extract_center_pts] → pts_meters [(X,Y),...]
          pts_meters → [LaneEstimator]   → (e_y, e_psi, valid)
          tất cả    → [_draw_debug]      → debug_img

        Parameters
        ----------
        cv_image : np.ndarray
            Ảnh BGR gốc từ camera, shape (H, W, 3), dtype uint8.
            Thường là 640×480 nhưng code không cứng nhắc kích thước.

        Returns
        -------
        e_y       : float       Sai số lệch ngang tại Ld=1.0m (m).
        e_psi     : float       Sai số góc hướng tại mũi xe (rad).
        mask      : np.ndarray  Mặt nạ nhị phân làn đường, shape (H, W).
        debug_img : np.ndarray  Ảnh BGR annotated để hiển thị / publish.
        """
        try:
            # ── Bước 1: AI phát hiện vạch làn ─────────────────────────────
            mask = self._detector.detect(cv_image)

            # ── Bước 2: Trích xuất điểm tâm làn (pixel → mét) ────────────
            pts_meters = self._extract_center_pts(mask)

            # ── Bước 3: Khớp đa thức và tính sai số điều khiển ───────────
            e_y, e_psi, valid = self._estimator.estimate(pts_meters)

            # Lấy hệ số đa thức hiện tại để vẽ đường cong debug
            coeffs = self._estimator._cached_coeffs if self._estimator.has_cache else None

            # ── Bước 4: Vẽ ảnh debug ──────────────────────────────────────
            debug_img = self._draw_debug(cv_image, mask, e_y, e_psi, valid, coeffs)

            return e_y, e_psi, mask, debug_img

        except Exception as exc:
            # An toàn: bất kỳ lỗi nào → trả về giá trị mặc định an toàn
            debug_img = cv_image.copy()
            cv2.putText(
                debug_img, f'ERR: {exc}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            return 0.0, 0.0, np.zeros(cv_image.shape[:2], dtype=np.uint8), debug_img


# ──────────────────────────────────────────────────────────────────────────────
# Chạy thẳng từ terminal:
#   python3 processor.py /path/to/video.mp4
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os

    if len(sys.argv) < 2:
        print('Cách dùng: python3 processor.py <video_file>')
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'EgoLanes_Lite_FP32.onnx')

    proc = LaneProcessor(model_path)
    cap  = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Không mở được video: {video_path}')
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        e_y, e_psi, _, debug = proc.process_frame(frame)
        print(f'e_y={e_y:+.4f}m  e_psi={math.degrees(e_psi):+.2f}deg')
        cv2.imshow('Lane Debug', debug)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
