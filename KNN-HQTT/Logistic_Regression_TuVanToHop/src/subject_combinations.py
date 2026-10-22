"""Cấu hình các tổ hợp môn chính thức do người dùng cung cấp."""

# Chưa có danh sách chính thức nên để trống, không tự suy đoán tên hoặc tổ hợp.
# Sau khi nhận dữ liệu chính xác, khai báo theo mẫu:
# OFFICIAL_COMBINATIONS = {
#     "Tổ hợp 1": ("Tên môn chính xác", "Tên môn chính xác"),
#     "Tổ hợp 2": ("Tên môn chính xác", "Tên môn chính xác"),
#     "Tổ hợp 3": ("Tên môn chính xác", "Tên môn chính xác"),
#     "Tổ hợp 4": ("Tên môn chính xác", "Tên môn chính xác"),
#     "Tổ hợp 5": ("Tên môn chính xác", "Tên môn chính xác"),
# }
OFFICIAL_COMBINATIONS: dict[str, tuple[str, ...]] = {}
