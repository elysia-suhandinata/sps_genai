import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.model import CIFARCNN


class ImageClassifier:
    def __init__(self, model_path="cifar_cnn.pth"):
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.model = CIFARCNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()
        return self.classes[predicted_index]