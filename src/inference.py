import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import DefectClassifier
import sys
import os

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/defect_classifier.pth"
LABELS     = {0: "GOOD", 1: "DEFECT"}
COLORS     = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green=good, Red=defect

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    model = DefectClassifier(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path):
    img_pil = Image.open(image_path).convert("RGB")
    tensor  = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    
    return pred.item(), conf.item()

def run_on_image(image_path):
    model  = load_model()
    label_idx, confidence = predict_image(model, image_path)
    label  = LABELS[label_idx]
    color  = COLORS[label_idx]
    
    # Display with OpenCV
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (512, 512))
    text   = f"{label}: {confidence*100:.1f}%"
    cv2.putText(img_cv, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Defect Inspection", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Result: {label} ({confidence*100:.1f}% confidence)")

def run_on_folder(folder_path):
    model   = load_model()
    results = []
    
    for fname in os.listdir(folder_path):
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue
        fpath = os.path.join(folder_path, fname)
        label_idx, conf = predict_image(model, fpath)
        results.append((fname, LABELS[label_idx], conf))
        print(f"{fname}: {LABELS[label_idx]} ({conf*100:.1f}%)")
    
    defects = sum(1 for _, l, _ in results if l == "DEFECT")
    print(f"\nSummary: {defects}/{len(results)} defective parts detected")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path_or_folder>")
        sys.exit(1)
    
    path = sys.argv[1]
    if os.path.isdir(path):
        run_on_folder(path)
    else:
        run_on_image(path)