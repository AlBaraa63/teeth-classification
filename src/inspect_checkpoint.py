import torch
import os
import config

MODEL_PATH = os.path.join(config.OUTPUT_DIR, "best_model.pth")
print(f"Inspecting: {MODEL_PATH}")

try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    print("Keys in checkpoint:", checkpoint.keys())
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"Val Acc: {checkpoint['val_acc']}")
    
    state_dict = checkpoint['model_state_dict']
    print("First 5 keys in state_dict:")
    for i, key in enumerate(state_dict.keys()):
        if i >= 5: break
        print(f"  {key}")
        
    if 'initial.0.weight' in state_dict:
        print("\n=> DETECTED: SCOPE 'initial' found. This is the SCRATCH model.")
    elif 'conv1.weight' in state_dict:
        print("\n=> DETECTED: SCOPE 'conv1' found. This is likely RESNET.")
        
except Exception as e:
    print(f"Error loading: {e}")
