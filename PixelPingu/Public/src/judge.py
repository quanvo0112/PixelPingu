import torch
from torchvision.models import shufflenet_v2_x2_0, regnet_x_1_6gf
from torchvision.models import ShuffleNet_V2_X2_0_Weights, RegNet_X_1_6GF_Weights
from PIL import Image
import numpy as np
import os


class PenguinJudge:
    def __init__(self):
        self.judge_one_model = shufflenet_v2_x2_0(weights=None)
        self.judge_two_model = regnet_x_1_6gf(weights=None)

        self.load_custom_weights()

        self.judge_one_model.eval()
        self.judge_two_model.eval()

        self.judge_one_transform = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1.transforms()
        self.judge_two_transform = RegNet_X_1_6GF_Weights.IMAGENET1K_V2.transforms()

        self.penguin_class = 145

        self.flag = os.getenv(
            "FLAG", "HCMUS-CTF{FAKEEEEEE_FLAGGGGG_FAKEEEEEE_FLAGGGGG}"
        )
        self.flag_parts = self.split_flag_into_parts(self.flag)

    def split_flag_into_parts(self, flag):
        part_length = len(flag) // 4
        remainder = len(flag) % 4

        parts = []
        start = 0
        for i in range(4):
            extra = 1 if i < remainder else 0
            end = start + part_length + extra
            parts.append(flag[start:end])
            start = end

        return parts

    def get_flag_part(self, judge_one_is_penguin, judge_two_is_penguin):
        if not judge_one_is_penguin and not judge_two_is_penguin:
            return self.flag_parts[0]
        elif judge_one_is_penguin and judge_two_is_penguin:
            return self.flag_parts[1]
        elif judge_one_is_penguin and not judge_two_is_penguin:
            return self.flag_parts[2]
        elif not judge_one_is_penguin and judge_two_is_penguin:
            return self.flag_parts[3]

    def canvas_data_to_image(self, canvas_data):
        data_array = np.array(canvas_data, dtype=np.uint8)
        img_array = data_array.reshape((128, 128, 4))
        rgb_array = img_array[:, :, :3]
        return Image.fromarray(rgb_array, "RGB")

    def predict_single_model(self, image, model, transform):
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            return torch.softmax(outputs[0], dim=0)

    def score_penguin_artwork(self, canvas_data):
        try:
            image = self.canvas_data_to_image(canvas_data)

            judge_one_probs = self.predict_single_model(
                image, self.judge_one_model, self.judge_one_transform
            )
            judge_two_probs = self.predict_single_model(
                image, self.judge_two_model, self.judge_two_transform
            )

            judge_one_top_class = judge_one_probs.argmax().item()
            judge_two_top_class = judge_two_probs.argmax().item()

            judge_one_confidence = judge_one_probs[judge_one_top_class].item() * 100
            judge_two_confidence = judge_two_probs[judge_two_top_class].item() * 100

            judge_one_is_penguin = judge_one_top_class == self.penguin_class
            judge_two_is_penguin = judge_two_top_class == self.penguin_class

            score = 0
            if judge_one_is_penguin:
                score += judge_one_confidence
            if judge_two_is_penguin:
                score += judge_two_confidence

            score = (score / 200) * 100

            flag_part = self.get_flag_part(judge_one_is_penguin, judge_two_is_penguin)

            return {
                "score": score,
                "judge_one": {
                    "top_class": judge_one_top_class,
                    "confidence": judge_one_confidence,
                    "is_penguin": judge_one_is_penguin,
                },
                "judge_two": {
                    "top_class": judge_two_top_class,
                    "confidence": judge_two_confidence,
                    "is_penguin": judge_two_is_penguin,
                },
                "flag_part": flag_part,
            }
        except Exception as e:
            return {
                "score": 0,
                "error": str(e),
                "judge_one": {"top_class": -1, "confidence": 0, "is_penguin": False},
                "judge_two": {"top_class": -1, "confidence": 0, "is_penguin": False},
            }

    def load_custom_weights(self):
        try:
            if os.path.exists("models/shufflenet-weighs.pth"):
                self.judge_one_model.load_state_dict(
                    torch.load("models/shufflenet-weighs.pth", map_location="cpu")
                )

            if os.path.exists("models/regnet-weights.pth"):
                self.judge_two_model.load_state_dict(
                    torch.load("models/regnet-weights.pth", map_location="cpu")
                )
        except Exception as e:
            print(f"Error loading custom weights: {e}")


judge_instance = None


def get_judge_instance():
    global judge_instance
    if judge_instance is None:
        judge_instance = PenguinJudge()
    return judge_instance


def score_penguin_submission(canvas_data):
    return get_judge_instance().score_penguin_artwork(canvas_data)
