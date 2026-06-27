#!/usr/bin/env python3
"""
overtake_vision_node.py — Camera-based NPC detection bổ sung cho overtake pipeline.

Phát hiện NPC box màu cam (RGB 1.0, 0.45, 0.0) bằng HSV color filter.
Phân vùng ảnh thành:
  - Front sector: NPC đang ở phía trước trong làn robot (trigger overtake)
  - Adj sector  : NPC đang ở làn ngoài phía trước-trái (block overtake nếu không clear)

Publish:
  /overtake/vision_front_dist  (std_msgs/Float32)  — khoảng cách NPC phía trước [m], -1 = none
  /overtake/vision_adj_clear   (std_msgs/Bool)     — làn ngoài trống từ góc nhìn camera
  /overtake/vision_debug       (sensor_msgs/Image) — ảnh debug có bbox + sector lines

Subscribe:
  /camera/image_raw  (sensor_msgs/Image)
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Vector3

# ── Camera intrinsics (từ camera.xacro: FOV_H=120°, 640×480) ────────────────
_IMG_W = 640
_IMG_H = 480
_CX    = _IMG_W / 2.0                              # 320.0 px
_CY    = _IMG_H / 2.0                              # 240.0 px
_FX    = _IMG_W / (2.0 * math.tan(math.radians(60.0)))  # ≈ 184.75 px (fx=fy, square pixel)
_FY    = _FX

# ── NPC box dimensions (từ npc_driver_node.cpp) ──────────────────────────────
_NPC_WID = 0.190   # m — chiều rộng (Y-axis của NPC, +20% so với 0.158)
_NPC_HGT = 0.167   # m — chiều cao

# ── HSV range cho màu cam của NPC ────────────────────────────────────────────
# NPC ambient/diffuse = (1.0, 0.45, 0.0):
#   RGB(255,115,0) → HSV(H≈14°, S=100%, V=100%)
# OpenCV dùng H: 0–180, S: 0–255, V: 0–255
# Mở rộng để chịu thay đổi ánh sáng Gazebo
_HSV_LO = np.array([ 5, 110,  50], dtype=np.uint8)   # H=[5,25], S, V lower
_HSV_HI = np.array([25, 255, 255], dtype=np.uint8)

# ── ROI: loại bỏ bầu trời (trên) và mặt đường sát xe (dưới) ────────────────
_ROI_TOP = 80    # row trên cùng của ROI (bỏ ≈ top 17%)
_ROI_BOT = 420   # row dưới cùng của ROI (bỏ ≈ bottom 12.5%)

# ── Contour size filter ───────────────────────────────────────────────────────
_MIN_AREA = 25     # px² — loại nhiễu nhỏ
_MAX_AREA = 9000   # px² — NPC có area lớn nhất ≈ (30px × 20px) = 600px² tại 2m

# ── Phân loại vị trí (geometric) theo tọa độ robot ──────────────────────────
# - Front (làn trong — cùng làn robot): |Y_robot| < _FRONT_Y_MAX
# - Adj (làn ngoài — bên trái robot):   _ADJ_Y_MIN < Y_robot < _ADJ_Y_MAX
# Outer lane offset so với inner: 2.801 - 2.267 = 0.534m
_FRONT_Y_MAX = 0.40   # m — NPC trong làn đang chạy (±0.40m từ center)
_ADJ_Y_MIN   = 0.15   # m — NPC bắt đầu vào vùng làn ngoài
_ADJ_Y_MAX   = 1.10   # m — NPC ở rìa ngoài của làn ngoài

# ── EMA (Exponential Moving Average) ─────────────────────────────────────────
_ALPHA_DIST  = 0.40   # tốc độ cập nhật distance (0=không đổi, 1=tức thời)
_ALPHA_ADJ   = 0.45   # tốc độ cập nhật adj_clear score
_ADJ_THRESH  = 0.5    # score adj > 0.5 → làn ngoài được coi là trống

# ── Timeout ───────────────────────────────────────────────────────────────────
_IMG_TIMEOUT_S = 1.5   # giây — không nhận ảnh quá lâu → reset về mặc định

# ── Limits hợp lý cho distance estimate ─────────────────────────────────────
_DIST_MIN = 0.25   # m — gần hơn thì estimate không tin cậy
_DIST_MAX = 8.0    # m — xa hơn thì NPC bbox quá nhỏ (< 4px)


class OvertakeVisionNode(Node):
    """
    Phát hiện NPC bằng camera. Publish khoảng cách front NPC và trạng thái adj lane.

    Architecture:
        Layer 0: HSV filter → binary orange mask
        Layer 1: Morphology + contour detection → bounding boxes
        Layer 2: Geometric classification → front / adj lane
        Layer 3: Distance estimation từ bbox size (NPC width & height known)
        Layer 4: EMA smoothing chống oscillation
    """

    def __init__(self):
        super().__init__('overtake_vision_node')

        self.declare_parameter('image_topic',   '/camera/image_raw')
        self.declare_parameter('publish_debug', True)

        image_topic   = self.get_parameter('image_topic').value
        self._pub_dbg = self.get_parameter('publish_debug').value

        # Curvature từ lane_follower (κ = vyaw/vx) để bù góc nhìn camera trên curve
        self._kappa: float = 0.0

        # EMA state
        self._dist_ema: float  = -1.0   # -1 = không phát hiện
        self._adj_score: float =  1.0   # 1.0 = clear, 0.0 = occupied
        self._has_init: bool   = False
        self._last_img_t: float = 0.0

        # QoS sensor
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe
        self._sub = self.create_subscription(
            Image, image_topic, self._img_cb, sensor_qos)

        # Kappa từ /status_err.z (lane_follower publish κ = vyaw/vx)
        self.create_subscription(
            Vector3, '/status_err',
            lambda msg: setattr(self, '_kappa', float(msg.z)),
            10)

        # Publish
        self._pub_dist  = self.create_publisher(Float32, '/overtake/vision_front_dist', 10)
        self._pub_adj   = self.create_publisher(Bool,   '/overtake/vision_adj_clear',  10)
        self._pub_debug = self.create_publisher(Image,  '/overtake/vision_debug', 10)

        # Watchdog + publish timer ở 10 Hz
        self.create_timer(0.10, self._timer_cb)

        self.get_logger().info(
            f'[overtake_vision] ready — topic={image_topic}'
            f'  FX={_FX:.1f}px  ADJ_offset=0.534m'
        )

    # ── Image decode (không dùng cv_bridge) ──────────────────────────────────
    @staticmethod
    def _to_bgr(msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        nc = {'rgb8': 3, 'bgr8': 3, 'r8g8b8': 3,
              'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc, 3)
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, nc)
        if enc in ('rgb8', 'r8g8b8'):
            return frame[:, :, ::-1].copy()
        if enc == 'rgba8':
            return frame[:, :, 2::-1].copy()
        if enc == 'bgra8':
            return frame[:, :, :3].copy()
        return frame.copy()

    # ── Layer 3: Distance estimation ──────────────────────────────────────────
    @staticmethod
    def _estimate_dist(bbox_w: float, bbox_h: float) -> float:
        """
        D = f * L_real / L_px  (pinhole model)
        Trung bình có trọng số: ưu tiên height (ít bị cắt ngang hơn width).
        """
        d_w = _FX * _NPC_WID / max(bbox_w, 1.0)
        d_h = _FY * _NPC_HGT / max(bbox_h, 1.0)
        return (d_w + 1.5 * d_h) / 2.5

    # ── Layer 2: Geometric classification ────────────────────────────────────
    @staticmethod
    def _lateral_y(col_px: float, dist_m: float, kappa: float = 0.0) -> float:
        """
        Chuyển column pixel → lateral Y trong frame robot.
        Y > 0 = bên trái robot (= outer lane direction).

        Bù curve: trên đường cong κ, NPC cách D phía trước xuất hiện lệch
        FX·D·κ pixel về phía cong trong ảnh → trừ để lấy Y thực tế.
        """
        col_corrected = col_px - _FX * dist_m * kappa
        return (_CX - col_corrected) * dist_m / _FX

    # ── Main image callback ───────────────────────────────────────────────────
    def _img_cb(self, msg: Image):
        self._last_img_t = self.get_clock().now().nanoseconds * 1e-9

        try:
            bgr = self._to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f'[overtake_vision] decode error: {exc}')
            return

        h_img, w_img = bgr.shape[:2]

        # ── Layer 0: HSV filter trong ROI ────────────────────────────────
        roi = bgr[_ROI_TOP:_ROI_BOT, :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)

        # ── Layer 1: Morphology + contour detection ───────────────────────
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        front_dists: list[float] = []
        adj_occupied = False
        debug_rects  = []   # (x, y, w, h, color, label)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _MIN_AREA or area > _MAX_AREA:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx_box = bx + bw / 2.0          # column của tâm bbox trong ROI
            col_img = cx_box                 # column trong ảnh gốc (ROI không crop cột)
            row_img = by + _ROI_TOP + bh / 2.0

            # ── Layer 3: Distance ─────────────────────────────────────────
            d = self._estimate_dist(float(bw), float(bh))
            if not (_DIST_MIN < d < _DIST_MAX):
                continue

            # ── Layer 2: Lateral position (bù curve) ─────────────────────
            Y = self._lateral_y(col_img, d, self._kappa)

            # Phân loại: front (làn trong) vs adj (làn ngoài)
            is_front = abs(Y) < _FRONT_Y_MAX
            is_adj   = _ADJ_Y_MIN < Y < _ADJ_Y_MAX

            if is_front:
                front_dists.append(d)
                debug_rects.append((
                    bx, by + _ROI_TOP, bw, bh,
                    (0, 80, 255),        # đỏ-cam = front NPC
                    f'F {d:.1f}m Y={Y:+.2f}',
                ))

            if is_adj:
                adj_occupied = True
                debug_rects.append((
                    bx, by + _ROI_TOP, bw, bh,
                    (255, 80, 0),        # xanh dương = adj NPC
                    f'A {d:.1f}m Y={Y:+.2f}',
                ))

        # ── Layer 4: EMA update ───────────────────────────────────────────
        # Front distance
        raw_dist = min(front_dists) if front_dists else -1.0

        if raw_dist > 0:
            if not self._has_init:
                self._dist_ema = raw_dist
                self._has_init = True
            else:
                self._dist_ema = (_ALPHA_DIST * raw_dist
                                  + (1.0 - _ALPHA_DIST) * self._dist_ema)
        else:
            if self._dist_ema > 0:
                # Exponential decay khi mất dấu tạm thời
                self._dist_ema *= (1.0 - _ALPHA_DIST)
                if self._dist_ema < _DIST_MIN:
                    self._dist_ema = -1.0
                    self._has_init = False
            else:
                self._dist_ema = -1.0

        # Adj clear score: 1.0 = clear, 0.0 = occupied
        raw_adj = 0.0 if adj_occupied else 1.0
        self._adj_score = (_ALPHA_ADJ * raw_adj
                           + (1.0 - _ALPHA_ADJ) * self._adj_score)

        # ── Debug visualization ───────────────────────────────────────────
        if self._pub_dbg:
            self._publish_debug(bgr, mask, debug_rects, msg.header.stamp)

    # ── Debug image builder ───────────────────────────────────────────────────
    def _publish_debug(self, bgr, mask_roi, rects, stamp):
        dbg = bgr.copy()

        # Hiển thị mask (tô xanh nhạt lên vùng orange detected)
        mask_full = np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask_full[_ROI_TOP:_ROI_BOT, :] = mask_roi
        green_layer = np.zeros_like(dbg)
        green_layer[:, :, 1] = 100
        dbg = np.where(mask_full[:, :, None] > 0, green_layer, dbg).astype(np.uint8)

        # ROI boundary
        cv2.rectangle(dbg,
                      (0, _ROI_TOP), (dbg.shape[1]-1, _ROI_BOT),
                      (150, 150, 150), 1)

        # Sector lines: đường phân chia front Y và adj Y trên ảnh (tại D=3m)
        # Tại D=3m: col = CX - FX * Y / D
        col_f_l = int(_CX + _FX * _FRONT_Y_MAX / 3.0)   # front right edge
        col_f_r = int(_CX - _FX * _FRONT_Y_MAX / 3.0)   # front left edge
        col_a_l = int(_CX - _FX * _ADJ_Y_MIN  / 3.0)    # adj right edge
        col_a_r = int(_CX - _FX * _ADJ_Y_MAX  / 3.0)    # adj left edge

        for cx, clr, lbl in [
            (col_f_l, (0, 200, 80),   'F-R'),
            (col_f_r, (0, 200, 80),   'F-L'),
            (col_a_l, (200, 150, 0),  'A-R'),
            (col_a_r, (200, 150, 0),  'A-L'),
        ]:
            if 0 < cx < dbg.shape[1]:
                cv2.line(dbg, (cx, _ROI_TOP), (cx, _ROI_BOT), clr, 1)
                cv2.putText(dbg, lbl, (cx-10, _ROI_TOP+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)

        # Bounding boxes
        for (bx, by, bw, bh, clr, lbl) in rects:
            cv2.rectangle(dbg, (bx, by), (bx+bw, by+bh), clr, 2)
            cv2.putText(dbg, lbl, (bx, max(by-4, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

        # Status text
        d_str  = f'{self._dist_ema:.2f}m' if self._dist_ema > 0 else 'none'
        adj_ok = self._adj_score > _ADJ_THRESH
        adj_str = f'CLR({self._adj_score:.2f})' if adj_ok else f'OCC({self._adj_score:.2f})'
        cv2.putText(dbg, f'front={d_str}  adj={adj_str}  k={self._kappa:+.2f}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Publish
        out = Image()
        out.header.stamp    = stamp
        out.header.frame_id = 'camera_link_optical'
        out.height    = dbg.shape[0]
        out.width     = dbg.shape[1]
        out.encoding  = 'bgr8'
        out.is_bigendian = False
        out.step      = dbg.shape[1] * 3
        out.data      = dbg.tobytes()
        self._pub_debug.publish(out)

    # ── Watchdog + publish timer ──────────────────────────────────────────────
    def _timer_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # Timeout: không nhận ảnh quá lâu → reset về trạng thái safe
        if self._last_img_t > 0 and (now - self._last_img_t) > _IMG_TIMEOUT_S:
            if self._dist_ema > 0 or self._adj_score < 1.0:
                self.get_logger().warning('[overtake_vision] image timeout — reset')
            self._dist_ema  = -1.0
            self._adj_score =  1.0
            self._has_init  = False

        # Publish
        dist_msg      = Float32()
        dist_msg.data = float(self._dist_ema)
        self._pub_dist.publish(dist_msg)

        adj_msg      = Bool()
        adj_msg.data = bool(self._adj_score > _ADJ_THRESH)
        self._pub_adj.publish(adj_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OvertakeVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
