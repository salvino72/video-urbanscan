import torch
import openvino as ov

print("--- Verifica Hardware ---")

# Verifica CUDA (NVIDIA)
cuda_available = torch.cuda.is_available()
print(f"CUDA Disponibile (NVIDIA): {cuda_available}")
if cuda_available:
    print(f"GPU NVIDIA: {torch.cuda.get_device_name(0)}")
    print(f"Numero di GPU: {torch.cuda.device_count()}")

# Verifica OpenVINO (Intel/CPU)
core = ov.Core()
devices = core.available_devices
print(f"Dispositivi OpenVINO rilevati: {devices}")
