from modelscope import AutoModelForSeq2SeqLM, AutoProcessor

processor = AutoProcessor.from_pretrained("ZhipuAI/GLM-ASR-Nano-2512")
model = AutoModelForSeq2SeqLM.from_pretrained("ZhipuAI/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(decoded_outputs)