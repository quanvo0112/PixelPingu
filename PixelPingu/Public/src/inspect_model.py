# inspect_models.py
import torch
from torchvision.models import shufflenet_v2_x2_0, regnet_x_1_6gf
import os
from judge import PenguinJudge 

judge = PenguinJudge()
penguin_class_id = judge.penguin_class

# 1. Khởi tạo kiến trúc model rỗng (chưa có trọng số huấn luyện)
#    Đây là các lớp kiến trúc y hệt trong judge.py
model_shufflenet = shufflenet_v2_x2_0(weights=None)
model_regnet = regnet_x_1_6gf(weights=None)

# 2. Nạp state_dict từ các file .pth
#    torch.load sẽ đọc file và trả về đối tượng Python đã được lưu
print("[*] Loading state dictionaries from .pth files...")
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one directory to reach the parent directory of src
parent_dir = os.path.dirname(current_dir)
# Define paths to model files
shufflenet_path = os.path.join(parent_dir, "src/models", "shufflenet-weighs.pth")
regnet_path = os.path.join(parent_dir, "src/models", "regnet-weights.pth")

# Load the models using the proper paths
shufflenet_state_dict = torch.load(shufflenet_path, map_location="cpu")
regnet_state_dict = torch.load(regnet_path, map_location="cpu")

# 3. Áp các trọng số đã nạp vào kiến trúc model
model_shufflenet.load_state_dict(shufflenet_state_dict)
model_regnet.load_state_dict(regnet_state_dict)

print("[+] Models loaded successfully!")

# inspect_models.py (tiếp theo)
from torchinfo import summary
import os

# Kích thước đầu vào mẫu (batch_size, channels, height, width)
# Phải giống với kích thước ảnh sau khi qua transform (thường là 224x224 cho ImageNet models)
input_size = (1, 3, 224, 224) 

print("\n" + "="*50)
print("              ShuffleNet V2 x2.0 Architecture")
print("="*50)
# In ra cấu trúc của model thứ nhất
summary(model_shufflenet, input_size=input_size)


print("\n" + "="*50)
print("               RegNet_X_1.6GF Architecture")
print("="*50)
# In ra cấu trúc của model thứ hai
summary(model_regnet, input_size=input_size)

# inspect_models.py (tiếp theo)

# So sánh trọng số của tầng tuyến tính (fully connected) cuối cùng
# Đây là tầng quyết định lớp đầu ra cuối cùng

# Lấy ra tầng fc của shufflenet
fc_shufflenet = model_shufflenet.fc.weight
# Lấy ra tầng fc của regnet
fc_regnet = model_regnet.fc.weight

print("\n" + "="*50)
print("            Comparing Final Layer Weights")
print("="*50)

print(f"Shape of ShuffleNet FC weights: {fc_shufflenet.shape}")
print(f"Shape of RegNet FC weights:     {fc_regnet.shape}")

# Tính toán sự khác biệt tuyệt đối trung bình giữa các trọng số
# Lưu ý: Điều này chỉ mang tính tham khảo vì hai kiến trúc khác nhau
# nên không thể so sánh trực tiếp. Nhưng ta có thể so sánh độ lớn của chúng.

print(f"\nAverage absolute weight value in ShuffleNet FC: {torch.mean(torch.abs(fc_shufflenet)):.6f}")
print(f"Average absolute weight value in RegNet FC:     {torch.mean(torch.abs(fc_regnet)):.6f}")

# So sánh trọng số liên quan đến lớp chim cánh cụt (145)
penguin_weights_shufflenet = fc_shufflenet[penguin_class_id]
penguin_weights_regnet = fc_regnet[penguin_class_id]

print(f"\nAverage absolute weight for PENGUIN CLASS (145) in ShuffleNet: {torch.mean(torch.abs(penguin_weights_shufflenet)):.6f}")
print(f"Average absolute weight for PENGUIN CLASS (145) in RegNet:     {torch.mean(torch.abs(penguin_weights_regnet)):.6f}")

