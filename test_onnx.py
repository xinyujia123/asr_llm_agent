from funasr import AutoModel
import librosa
import os

# Paths to exported models
OUTPUT_ROOT = "./exported_onnx_models"
ASR_MODEL_PATH = os.path.join(OUTPUT_ROOT, "asr")
VAD_MODEL_PATH = os.path.join(OUTPUT_ROOT, "vad")
PUNC_MODEL_PATH = os.path.join(OUTPUT_ROOT, "punc")

# Verify paths exist
if not os.path.exists(ASR_MODEL_PATH):
    print(f"Error: ASR model path not found at {ASR_MODEL_PATH}. Please run export_onnx.py first.")
    exit(1)

print("Loading ONNX models...")

# When loading ONNX models with AutoModel:
# We pass the directory path containing the ONNX files.
# AutoModel should detect it's a local path.
# We might need to specify device="cuda" if we want GPU inference (requires onnxruntime-gpu).

try:
    asr_model = AutoModel(
        model=ASR_MODEL_PATH,
        vad_model=VAD_MODEL_PATH,
        punc_model=PUNC_MODEL_PATH,
        device="cuda", # Change to "cpu" if you don't have GPU or onnxruntime-gpu
        disable_update=True,
        trust_remote_code=False,
        use_itn=False
    )
    
    print(f"Models loaded successfully. Device: {asr_model.device}")
    
    # Test inference
    raw_path = "/workspace/audio_llm_agent/test/test_audio/1.amr"
    # Note: Adjust path if needed, e.g. using relative path or checking existence
    if not os.path.exists(raw_path):
        # Fallback to a dummy file or user's file path if the above doesn't exist
        # But let's assume the user has the file from the original test_export.py
        # Try to find it in current project if absolute path fails
        local_path = "./test/test_audio/1.amr"
        if os.path.exists(local_path):
            raw_path = local_path
        else:
            print(f"Warning: Test audio file not found at {raw_path}")

    if os.path.exists(raw_path):
        print(f"Running inference on {raw_path}...")
        audio, _ = librosa.load(raw_path, sr=16000)
        
        # generate() usage should be the same
        res = asr_model.generate(input=audio)
        print(f"Result: {res}")
    else:
        print("Skipping inference test (audio file not found).")

except Exception as e:
    print(f"Failed to load or run ONNX models: {e}")
    import traceback
    traceback.print_exc()