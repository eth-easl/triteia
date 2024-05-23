from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
from naming import hf_name_to_path_friendly_name
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantized_model_dir = f".local/{hf_name_to_path_friendly_name(pretrained_model_dir)}"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
    # is_marlin_format=True,
)
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
model.save_quantized(quantized_model_dir, use_safetensors=True)
model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir, device="cuda:0",
    use_marlin=False,
)
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
