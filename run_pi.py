import os
import time
from ultralytics import YOLO

print("=== KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN TRÊN RASPBERRY PI 4 ===")

# Đường dẫn tới thư mục NCNN và thư mục ảnh test TRÊN RASPBERRY PI
# Đổi lại đường dẫn này cho khớp với chỗ bạn vừa copy file sang Pi
duong_dan_model = '/home/rinaka/Desktop/Drone_Project/Bao_Cao_YOLOv7_Hybrid_BW/kaggle/working/Drone_YOLOv7_Hybrid_BW/train_v7_mobilenet_bw/weights/best_ncnn_model'
thu_muc_test = '/home/rinaka/Desktop/RGBT_Dataset/images'
thu_muc_luu = '/home/rinaka/Desktop/Drone_Project/ket_qua'

os.makedirs(thu_muc_luu, exist_ok=True)

# Nạp mô hình NCNN
print("\nĐang nạp mô hình NCNN...")
model = YOLO(duong_dan_model, task='detect')
print("[OK] Đã nạp thành công!")

# Quét danh sách ảnh test
danh_sach_anh = [f for f in os.listdir(thu_muc_test) if f.lower().endswith(('.jpg', '.png'))]
tong_thoi_gian = 0

print(f"\nBắt đầu chạy nhận diện trên {len(danh_sach_anh)} ảnh...")

# Làm nóng CPU Pi 4
print("Đang làm nóng CPU...")
if len(danh_sach_anh) > 0:
    _ = model.predict(source=os.path.join(thu_muc_test, danh_sach_anh[0]), imgsz=640, verbose=False)

# Chạy vòng lặp nhận diện
for ten_anh in danh_sach_anh:
    duong_dan_anh = os.path.join(thu_muc_test, ten_anh)
    
    start = time.time()
    # Chạy inference
    results = model.predict(source=duong_dan_anh, imgsz=640, conf=0.3, verbose=False)
    end = time.time()
    
    thoi_gian_xu_ly = (end - start) * 1000
    tong_thoi_gian += thoi_gian_xu_ly
    
    # Lưu ảnh có vẽ khung nhận diện
    duong_dan_luu = os.path.join(thu_muc_luu, ten_anh)
    results[0].save(duong_dan_luu)
    
    print(f" - Xử lý {ten_anh:<20} | Tốc độ: {thoi_gian_xu_ly:.2f} ms")

# Tổng kết
if len(danh_sach_anh) > 0:
    fps_trung_binh = 1000 / (tong_thoi_gian / len(danh_sach_anh))
    print("\n" + "="*40)
    print(f"Tốc độ khung hình trung bình : {fps_trung_binh:.2f} FPS")
    print(f"Thời gian xử lý trung bình   : {(tong_thoi_gian / len(danh_sach_anh)):.2f} ms/ảnh")
    print(f"Ảnh kết quả đã được lưu tại  : {thu_muc_luu}")
    print("="*40)