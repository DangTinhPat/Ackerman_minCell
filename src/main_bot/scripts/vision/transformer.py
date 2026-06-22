#!/usr/bin/env python3
"""
transformer.py — Khối Hình Học Camera & Ánh Xạ Phối Cảnh Ngược (IPM).

Quản lý ma trận nội tại K và thực hiện phép chiếu hai chiều:
  - pixel_to_vehicle : pixel ảnh gốc → tọa độ mặt đất (mét, hệ ROS)
  - vehicle_to_pixel : tọa độ mặt đất (mét) → pixel ảnh gốc (dùng để vẽ debug)

KHÔNG phụ thuộc bất kỳ thư viện ROS 2 nào.

Quy ước hệ trục ROS (REP-103):
  X  — hướng tiến về phía trước xe  (dương = phía trước)
  Y  — hướng sang trái              (dương = bên trái)
  Z  — hướng lên trên              (dương = lên)

Thông số camera (từ robot_core.xacro & camera.xacro):
  Độ phân giải       : 640 × 480 px
  Horizontal FOV     : 1.089 rad
  Chiều cao lens     : 0.134 m so với mặt đất
  Offset dọc theo X  : 0.1485 m (camera đặt ở mặt trước thân xe)
  Góc nghiêng (Pitch): 0.0 rad  (nhìn thẳng, rpy="0 0 0")
"""

import math
from typing import Optional, Tuple

import numpy as np
import cv2

# ── Giới hạn vùng IPM hợp lệ ──────────────────────────────────────────────────
_MIN_X_M = 0.05   # m — bỏ qua vùng ngay dưới camera (nhiễu hình học)
_MAX_X_M = 4.0    # m — bỏ qua các điểm chiếu xa quá (không ổn định)


class GeometryTransformer:
    """
    Thực hiện ánh xạ phối cảnh ngược (IPM — Inverse Perspective Mapping)
    phân tích theo mô hình camera pinhole phẳng (flat-ground pinhole model).

    Ma trận nội tại K được tính từ độ phân giải và FOV ngang:
        fx = W / (2 · tan(h_fov / 2))
        fy = fx   (giả định pixel vuông)
        cx = W / 2,  cy = H / 2

    Phép chiếu thuận (pixel → mặt đất) giải bài toán giao tia – mặt phẳng:
        ray = K⁻¹ · [u, v, 1]ᵀ  (trong hệ camera)
        → đổi sang hệ xe → tìm t sao cho điểm trên tia có z = 0.
    """

    def __init__(
        self,
        img_w:       int   = 640,
        img_h:       int   = 480,
        h_fov:       float = 1.089,    # rad, horizontal FOV
        cam_height:  float = 0.134,    # m, chiều cao camera so với mặt đất
        cam_pitch:   float = 0.0,      # rad, góc cúi xuống (dương = cúi về phía đường)
        cam_x_offset: float = 0.1485,  # m, khoảng cách từ tâm xe đến camera theo X
    ):
        self.img_w        = img_w
        self.img_h        = img_h
        self.cam_height   = cam_height
        self.cam_pitch    = cam_pitch
        self.cam_x_offset = cam_x_offset

        # ── Tính ma trận nội tại K từ FOV và độ phân giải ─────────────────
        # fx = f_pixel theo trục X:  tan(hfov/2) = (W/2) / fx  →  fx = W/(2·tan(hfov/2))
        self.fx = img_w / (2.0 * math.tan(h_fov / 2.0))
        self.fy = self.fx          # pixel vuông → fy = fx
        self.cx = img_w / 2.0     # điểm chính (principal point) theo chiều ngang
        self.cy = img_h / 2.0     # điểm chính theo chiều dọc

        # ── Tiền tính metadata IPM cho từng hàng hợp lệ ───────────────────
        # Mỗi hàng v lưu: tỉ lệ mét/pixel theo chiều ngang tại hàng đó.
        # Dùng để ước tính chiều rộng làn khi chỉ thấy một vạch.
        self._row_lateral_scale: dict = {}  # v (int) → mét/pixel (float)
        self._precompute_row_scales()

    # ──────────────────────────────────────────────────────────────────────────
    # Tiền tính: tỉ lệ mét/pixel theo chiều ngang cho mỗi hàng ảnh
    # ──────────────────────────────────────────────────────────────────────────
    def _precompute_row_scales(self):
        """
        Với mỗi hàng v bên dưới đường chân trời, tính:
            lateral_scale[v] = |ΔY_mét| khi u thay đổi 1 pixel

        Dùng để đổi "nửa chiều rộng làn giả định (mét)" → "số pixel"
        trong trường hợp chỉ nhìn thấy một vạch làn.
        """
        for v in range(int(self.cy) + 1, self.img_h):
            pt_center = self._project_raw(self.cx,        float(v))
            pt_side   = self._project_raw(self.cx + 1.0, float(v))
            if pt_center is None or pt_side is None:
                continue
            scale = abs(pt_side[1] - pt_center[1])  # ΔY / 1px
            if scale > 1e-7:
                self._row_lateral_scale[v] = scale

    # ──────────────────────────────────────────────────────────────────────────
    # Chiếu thuận nội bộ: pixel → mặt đất (không giới hạn phạm vi)
    # ──────────────────────────────────────────────────────────────────────────
    def _project_raw(
        self, u: float, v: float
    ) -> Optional[Tuple[float, float]]:
        """
        Chiếu tia từ pixel (u, v) đến giao điểm với mặt phẳng mặt đất (z=0).

        Mô hình hình học (pinhole + phẳng đất):
        ─────────────────────────────────────────
        1. Chuẩn hoá pixel về hệ camera:
               du = (u - cx) / fx     ← offset góc theo trục ngang
               dv = (v - cy) / fy     ← offset góc theo trục dọc

        2. Chuyển hướng tia từ hệ camera sang hệ xe (dùng ma trận xoay pitch p):
               r_x = cos(p) − sin(p)·dv   ← thành phần tiến (forward)
               r_y = −du                   ← thành phần ngang (lateral)
               r_z = −sin(p) − cos(p)·dv  ← thành phần đứng (vertical, âm = xuống)

           Lưu ý "đảo dấu" r_y = −du:
               Pixel lệch phải (u > cx → du > 0) → Y_xe âm (sang phải) ✓
               Pixel lệch trái (u < cx → du < 0) → Y_xe dương (sang trái) ✓

        3. Tìm tham số t trên tia sao cho điểm giao cắt mặt đất:
               Camera ở độ cao h → giao với z=0 khi: h + t·r_z = 0
               → t = h / (−r_z)   (chỉ hợp lệ khi r_z < 0, tức tia nhìn xuống)

        4. Tính tọa độ giao điểm trong hệ xe:
               X = cam_x_offset + t·r_x    (tiến về phía trước)
               Y = t·r_y                   (sang trái, dương)
        """
        p  = self.cam_pitch
        h  = self.cam_height
        du = (u - self.cx) / self.fx
        dv = (v - self.cy) / self.fy

        # Thành phần thẳng đứng của tia trong hệ xe
        # (âm = tia hướng xuống đất; dương/zero = tia hướng lên/ngang → không giao đất)
        r_veh_z = -math.sin(p) - math.cos(p) * dv
        if r_veh_z >= -1e-6:
            return None  # Tia hướng lên hoặc song song với đất → không có giao điểm

        # Tham số khoảng cách dọc theo tia đến mặt đất
        t = h / (-r_veh_z)

        # Thành phần tiến (forward) của tia trong hệ xe
        r_veh_x = math.cos(p) - math.sin(p) * dv

        X = self.cam_x_offset + t * r_veh_x   # tọa độ tiến (m)
        Y = t * (-du)                           # tọa độ ngang (m, dương = trái)
        return X, Y

    # ──────────────────────────────────────────────────────────────────────────
    # API công khai — Chiếu thuận: pixel → mét (có lọc phạm vi)
    # ──────────────────────────────────────────────────────────────────────────
    def pixel_to_vehicle(
        self, px: float, py: float
    ) -> Optional[Tuple[float, float]]:
        """
        Đổi tọa độ pixel ảnh gốc (640×480) sang tọa độ vật lý mét trên mặt đất.

        Parameters
        ----------
        px : float  Cột pixel (u), gốc ở góc trên bên trái.
        py : float  Hàng pixel (v), tăng xuống dưới.

        Returns
        -------
        (X_m, Y_m) : Tọa độ hệ xe theo chuẩn ROS
                     X — hướng tiến (m), Y — hướng trái (m).
        None       : Nếu pixel trên đường chân trời hoặc chiếu ra ngoài phạm vi.
        """
        pt = self._project_raw(px, py)
        if pt is None:
            return None
        X, Y = pt
        # Loại bỏ các điểm quá gần (dưới camera) hoặc quá xa (không ổn định)
        if not (_MIN_X_M < X < _MAX_X_M):
            return None
        return X, Y

    # ──────────────────────────────────────────────────────────────────────────
    # API công khai — Chiếu ngược: mét → pixel (dùng để vẽ debug)
    # ──────────────────────────────────────────────────────────────────────────
    def vehicle_to_pixel(
        self, mx: float, my: float
    ) -> Optional[Tuple[int, int]]:
        """
        Chiếu ngược tọa độ vật lý mét (hệ xe) về pixel trên ảnh gốc 640×480.

        Dùng để vẽ đường tâm làn dự báo, điểm kiểm tra, v.v. lên ảnh debug.

        Giải tích ngược từ phương trình _project_raw:
        ────────────────────────────────────────────
        Cho X, Y → tìm (u, v):

        1. Từ phương trình X:
               X − cam_x = h · (cos p − sin p · dv) / (sin p + cos p · dv)
               Đặt r = (X − cam_x) / h:
               → dv = (cos p − r · sin p) / (r · cos p + sin p)

        2. Từ dv → v = cy + dv · fy

        3. Tính lại t từ dv:
               r_veh_z = −sin p − cos p · dv
               t = h / (−r_veh_z)

        4. Từ Y và t:
               Y = t · (−du)  →  du = −Y / t
               u = cx + du · fx

        Parameters
        ----------
        mx : float  X trong hệ xe (m, tiến về phía trước).
        my : float  Y trong hệ xe (m, dương = sang trái).

        Returns
        -------
        (u, v) : Tọa độ pixel nguyên trên ảnh 640×480.
        None   : Nếu điểm nằm ngoài vùng nhìn hợp lệ.
        """
        p      = self.cam_pitch
        h      = self.cam_height
        cam_x  = self.cam_x_offset

        if abs(mx - cam_x) < 1e-6:
            return None  # Điểm trùng với vị trí camera → kỳ dị

        # Bước 1: Tính dv từ X
        r    = (mx - cam_x) / h
        den  = r * math.cos(p) + math.sin(p)
        if abs(den) < 1e-9:
            return None
        dv = (math.cos(p) - r * math.sin(p)) / den

        # Bước 2: Tính v; phải nằm trong nửa dưới (phần đường, dưới đường chân trời)
        v_f = self.cy + dv * self.fy
        if not (self.cy < v_f < self.img_h):
            return None

        # Bước 3: Tính lại t tại hàng này
        r_veh_z = -math.sin(p) - math.cos(p) * dv
        if r_veh_z >= -1e-6:
            return None
        t = h / (-r_veh_z)

        # Bước 4: Tính u từ Y
        # Y = t · (−du)  →  du = −Y / t  →  u = cx + du · fx
        u_f = self.cx - my * self.fx / t
        if not (0 <= u_f < self.img_w):
            return None

        return int(round(u_f)), int(round(v_f))

    # ──────────────────────────────────────────────────────────────────────────
    # Trợ lý: lấy tỉ lệ mét/pixel tại một hàng ảnh
    # ──────────────────────────────────────────────────────────────────────────
    def lateral_scale_at(self, row: int) -> Optional[float]:
        """
        Trả về tỉ lệ mét/pixel theo chiều ngang tại hàng `row`.

        Dùng để đổi "nửa chiều rộng làn giả định" từ mét sang pixel
        khi xử lý hàng ảnh chỉ nhìn thấy một vạch làn.

        Returns None nếu hàng đó không có dữ liệu IPM hợp lệ.
        """
        return self._row_lateral_scale.get(row)
