from funasr import AutoModel
import os
import shutil

# Define model IDs as in test_export.py
ASR_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

OUTPUT_ROOT = "./exported_onnx_models"

def export_model(model_id, model_revision, output_dir, model_name="model"):
    print(f"\n--- Exporting {model_name} ({model_id}) ---")
    if os.path.exists(output_dir):
        print(f"Directory {output_dir} exists, cleaning up...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the model specifically for export
        # We load each component separately to ensure clean export
        model = AutoModel(
            model=model_id,
            model_revision=model_revision,
            device="cpu", # Export on CPU is often more stable for ONNX conversion
            disable_update=True
        )
        
        # Export
        # quantize=False for FP32 export. You can set True for int8 quantization.
        print(f"Starting export to {output_dir}...")
        model.export(output_dir=output_dir, quantize=False)
        print(f"Successfully exported {model_name}")
        
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 1. Export ASR
    export_model(ASR_MODEL_ID, "v2.0.4", os.path.join(OUTPUT_ROOT, "asr"), "ASR")

    # 2. Export VAD
    export_model(VAD_MODEL_ID, "v2.0.4", os.path.join(OUTPUT_ROOT, "vad"), "VAD")

    # 3. Export PUNC
    export_model(PUNC_MODEL_ID, "v2.0.4", os.path.join(OUTPUT_ROOT, "punc"), "PUNC")

    print("\nAll exports attempted.")
    print(f"Check {OUTPUT_ROOT} for results.")

if __name__ == "__main__":
    main()