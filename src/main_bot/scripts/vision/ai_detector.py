#!/usr/bin/env python3
"""
ai_detector.py — Khối Suy Luận AI (ONNX Inference).

Chịu trách nhiệm duy nhất: nhận ảnh BGR thô → trả về mặt nạ nhị phân làn đường.
KHÔNG phụ thuộc bất kỳ thư viện ROS 2 nào — có thể chạy offline bằng video.

Model: EgoLanes_Lite_FP32.onnx
  Input  : [1, 3, H_model, W_model]  float32, ImageNet-normalised RGB
  Output : [1, 3, H_model, W_model]  logits (multi-label, NOT argmax)
             ch0 = vạch làn trái  (left  ego-lane)
             ch1 = vạch làn phải (right ego-lane)
             ch2 = làn khác      (ignored)
"""

import numpy as np
import cv2
import onnxruntime as ort

# ── Hệ số chuẩn hoá ImageNet ──────────────────────────────────────────────────
# Mô hình được huấn luyện trên ảnh đã chuẩn hoá theo thống kê ImageNet.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Chỉ số kênh đầu ra
_CH_LEFT  = 0   # kênh 0: vạch làn trái
_CH_RIGHT = 1   # kênh 1: vạch làn phải


class AIDetector:
    """
    Nạp mô hình ONNX và chạy phát hiện vạch làn đường trên từng khung hình.

    Kích thước đầu vào của mô hình được đọc động từ metadata của ONNX session,
    nên class này tương thích với bất kỳ variant nào của mô hình EgoLanes.
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Đường dẫn tuyệt đối tới file .onnx trên đĩa.
        """
        # Ưu tiên GPU (CUDA), tự động fallback về CPU nếu không khả dụng
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._sess     = ort.InferenceSession(model_path, providers=providers)
        self._inp_name = self._sess.get_inputs()[0].name

        # Đọc động kích thước đầu vào từ metadata ONNX (định dạng NCHW)
        _, _, self._model_h, self._model_w = self._sess.get_inputs()[0].shape

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: Tiền xử lý ảnh (Pre-processing)
    # ──────────────────────────────────────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi ảnh BGR gốc → tensor NCHW float32 chuẩn hoá ImageNet.

        Quy trình biến đổi:
          1. BGR → RGB   (mô hình huấn luyện với ảnh RGB PyTorch)
          2. Resize      → (model_w, model_h) bằng nội suy song tuyến tính
          3. Chuẩn hoá   → [0,1] rồi trừ mean / chia std của ImageNet
          4. Chuyển trục → HWC → CHW → NCHW (thêm chiều batch = 1)
        """
        # Bước 1: Đổi thứ tự kênh màu vì OpenCV dùng BGR, mô hình cần RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Bước 2: Thu nhỏ về kích thước đầu vào của mô hình
        resized = cv2.resize(
            rgb, (self._model_w, self._model_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Bước 3: Chuẩn hoá ImageNet — đưa phân phối pixel về miền mô hình đã học
        #   pixel_norm = (pixel / 255 - mean) / std
        img = resized.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        # Bước 4: Chuyển trục từ HWC (NumPy) sang NCHW (PyTorch/ONNX)
        #   (H, W, 3) → (3, H, W) → (1, 3, H, W)
        tensor = img.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: Chạy suy luận và hợp nhất mặt nạ (Inference + Mask Fusion)
    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Phát hiện vạch làn đường, trả về mặt nạ nhị phân tổng hợp.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Ảnh BGR gốc từ camera, shape (H_orig, W_orig, 3), dtype uint8.

        Returns
        -------
        mask : np.ndarray
            Mặt nạ nhị phân uint8, shape (H_orig, W_orig).
            Giá trị 1 = pixel thuộc vạch làn TRÁI hoặc vạch làn PHẢI.
            Kích thước bằng đúng ảnh gốc đầu vào.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # ── Chạy suy luận ONNX ────────────────────────────────────────────
        tensor = self._preprocess(frame_bgr)
        # logits shape: [1, 3, H_model, W_model]
        logits = self._sess.run(None, {self._inp_name: tensor})[0]
        pred   = logits[0]  # [3, H_model, W_model]

        # ── Ngưỡng hoá tại logit = 0  ←→  sigmoid = 0.5 ──────────────────
        # Pixel nền / bầu trời có logit âm → tự động = 0 sau ngưỡng hoá.
        # Không cần softmax vì đây là bài toán phân loại đa nhãn (multi-label).
        left_mask  = (pred[_CH_LEFT]  > 0).astype(np.uint8)
        right_mask = (pred[_CH_RIGHT] > 0).astype(np.uint8)

        # ── Hợp nhất hai mặt nạ → một mặt nạ nhị phân duy nhất ──────────
        combined = np.bitwise_or(left_mask, right_mask)

        # ── Phóng to về kích thước ảnh gốc ───────────────────────────────
        # Dùng INTER_NEAREST để giữ nguyên giá trị nhị phân 0/1,
        # tránh nội suy tạo ra các giá trị trung gian.
        mask = cv2.resize(
            combined, (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        return mask
