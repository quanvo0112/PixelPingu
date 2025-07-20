# solver.py (phiên bản nâng cấp)

import torch
import requests
import numpy as np
from PIL import Image
from judge import PenguinJudge 

# --- KHỞI TẠO ---
print("[*] Initializing models and transforms...")
judge = PenguinJudge()
model1 = judge.judge_one_model
model2 = judge.judge_two_model
transform1 = judge.judge_one_transform
transform2 = judge.judge_one_transform # Lưu ý: Dùng cùng transform để so sánh dễ hơn, hoặc giữ nguyên transform2 nếu chúng khác nhau
penguin_class_id = judge.penguin_class

# <--- THAY ĐỔI 1: Sửa lại URL ---
CHALLENGE_URL = "http://103.199.17.56:25001/submit_artwork"

# <--- THAY ĐỔI 2: Nâng cấp hàm generate_adversarial_image ---
def generate_adversarial_image(
    target_m1_is_penguin, 
    target_m2_is_penguin, 
    steps=300, 
    lr=0.02,
    weight1=1.0,  # Trọng số ưu tiên cho mục tiêu của model 1
    weight2=1.0   # Trọng số ưu tiên cho mục tiêu của model 2
):
    """
    Tạo ảnh đối kháng với trọng số ưu tiên cho từng mục tiêu.
    """
    print(f"[*] Generating image for target: (Model1: {target_m1_is_penguin}, Model2: {target_m2_is_penguin}) with weights (w1={weight1}, w2={weight2})")
    
    image = torch.rand((1, 3, 128, 128), requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Áp dụng transform. Giữ nguyên như cũ.
        img_transformed1 = transform1(image)
        img_transformed2 = transform2(image)

        out1 = model1(img_transformed1)
        out2 = model2(img_transformed2)

        log_softmax1 = torch.nn.functional.log_softmax(out1, dim=1)
        log_softmax2 = torch.nn.functional.log_softmax(out2, dim=1)
        
        # Hàm loss cho từng model, giữ nguyên logic.
        loss1 = log_softmax1[0, penguin_class_id] if target_m1_is_penguin else -log_softmax1[0, penguin_class_id]
        loss2 = log_softmax2[0, penguin_class_id] if target_m2_is_penguin else -log_softmax2[0, penguin_class_id]

        # <--- THAY ĐỔI CỐT LÕI: Áp dụng trọng số ưu tiên ---
        # Mục tiêu vẫn là tối đa hóa (loss1 + loss2), nhưng giờ có trọng số.
        total_loss = -(weight1 * loss1 + weight2 * loss2)
        
        total_loss.backward()
        optimizer.step()
        image.data.clamp_(0, 1)

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{steps}, Loss: {total_loss.item():.4f}")

    return image.detach().squeeze(0)

def submit_image(image_tensor, url):
    """Hàm này không cần thay đổi."""
    img_np = image_tensor.numpy().transpose(1, 2, 0) * 255
    img_np = img_np.astype(np.uint8)
    rgba_img = np.concatenate([img_np, np.full((128, 128, 1), 255, dtype=np.uint8)], axis=-1)
    canvas_data = rgba_img.flatten().tolist()
    try:
        response = requests.post(url, json={"canvas_data": canvas_data}, timeout=30)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[!] Error submitting image: {e}")
        return None

if __name__ == "__main__":
    flag_parts = {}

    # --- Trường hợp 0: Cả hai đều KHÔNG thấy (lấy mảnh 0) ---
    print("\n--- Getting Flag Part 0 (Both Fail) ---")
    blank_tensor = torch.zeros(3, 128, 128)
    result0 = submit_image(blank_tensor, CHALLENGE_URL)
    print(f"    Server response: {result0}")
    if result0 and result0.get("success"):
        flag_parts[0] = result0.get("flag_part")

    # <--- THAY ĐỔI 3: Cập nhật các lệnh gọi hàm với trọng số ---
    # --- Trường hợp 1: Cả hai đều THẤY (lấy mảnh 1) ---
    # Không cần ưu tiên, cứ để cả hai cùng tiến.
    print("\n--- Getting Flag Part 1 (Both Pass) ---")
    image1 = generate_adversarial_image(target_m1_is_penguin=True, target_m2_is_penguin=True)
    result1 = submit_image(image1, CHALLENGE_URL)
    print(f"    Server response: {result1}")
    if result1 and result1.get("success"):
        flag_parts[1] = result1.get("flag_part")

    # --- Trường hợp 2: Model 1 THẤY, Model 2 KHÔNG (lấy mảnh 2) ---
    # Ưu tiên việc "THẤY" của model 1, vì "KHÔNG THẤY" thường dễ hơn.
    print("\n--- Getting Flag Part 2 (M1 Pass, M2 Fail) ---")
    image2 = generate_adversarial_image(
        target_m1_is_penguin=True, target_m2_is_penguin=False, 
        weight1=1.5, weight2=1.0  # Tập trung vào model 1
    )
    result2 = submit_image(image2, CHALLENGE_URL)
    print(f"    Server response: {result2}")
    if result2 and result2.get("success"):
        flag_parts[2] = result2.get("flag_part")

    # --- Trường hợp 3: Model 1 KHÔNG, Model 2 THẤY (lấy mảnh 3) ---
    # Ưu tiên việc "THẤY" của model 2.
    print("\n--- Getting Flag Part 3 (M1 Fail, M2 Pass) ---")
    image3 = generate_adversarial_image(
        target_m1_is_penguin=False, target_m2_is_penguin=True, 
        weight1=1.0, weight2=1.5  # Tập trung vào model 2
    )
    result3 = submit_image(image3, CHALLENGE_URL)
    print(f"    Server response: {result3}")
    if result3 and result3.get("success"):
        flag_parts[3] = result3.get("flag_part")

    # --- Ghép Flag ---
    if len(flag_parts) == 4:
        full_flag = "".join([flag_parts[i] for i in sorted(flag_parts.keys())])
        print(f"\n[+] SUCCESS! Full Flag: {full_flag}")
    else:
        print(f"\n[!] FAILED! Could not retrieve all flag parts. Got: {flag_parts}")