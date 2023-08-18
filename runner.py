from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
from pydantic import BaseModel
from fastapi import FastAPI
from typing import Union

app = FastAPI()
loaded = False
model_name = ''
mode = "local"

class LoaderItem(BaseModel):
    dir: str
    max_seq_len: Union[int, None]
    max_input_len: Union[int, None]
    max_attention_size: Union[int, None]
    compress_pos_emb: Union[float, None]
    alpha_value: Union[float, None]
    gpu_peer_fixed: Union[bool, None]
    auto_map: Union[bool, None]
    use_flash_attn_2: Union[bool, None]
    matmul_recons_thd: Union[int, None]
    fused_mlp_thd: Union[int, None]
    sdp_thd: Union[int, None]
    fused_attn: Union[bool, None]
    matmul_fused_remap: Union[bool, None]
    rmsnorm_no_half2: Union[bool, None]
    rope_no_half2: Union[bool, None]
    matmul_no_half2: Union[bool, None]
    silu_no_half2: Union[bool, None]
    concurrent_streams: Union[bool, None]

class GeneratorItem(BaseModel):
    temperature: Union[float, None]
    top_k: Union[int, None]
    top_p: Union[float, None]
    min_p: Union[float, None]
    typical: Union[float, None]
    token_repetition_penalty_max: Union[float, None]
    token_repetition_penalty_sustain: Union[int, None]
    token_repetition_penalty_decay: Union[int, None]
    beams: Union[int, None]
    beam_length: Union[int, None]
    disallowed_tokens: Union[list[int], None]
    prompt: str
    max_new_tokens: Union[int, None]


def loader(item: LoaderItem):
    # Directory containing model, tokenizer, generator
    dir = item.dir
    model_directory =  dir

    # set model name
    global model_name
    model_name = os.path.basename(dir)

    # Locate files we need within that directory

    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Create config, model, tokenizer and generator

    config = ExLlamaConfig(model_config_path)               # create config from config.json
    config.model_path = model_path                          # supply path to model weights file
    if item.max_seq_len is not None:
        config.max_seq_len = item.max_seq_len
    if item.max_input_len is not None:
        config.max_input_len = item.max_input_len
    if item.max_attention_size is not None:
        config.max_attention_size = item.max_attention_size
    if item.compress_pos_emb is not None:
        config.compress_pos_emb = item.compress_pos_emb
    if item.alpha_value is not None:
        config.alpha_value = item.alpha_value
    if item.gpu_peer_fixed is not None:
        config.gpu_peer_fixed = item.gpu_peer_fixed
    if item.auto_map is not None:
        config.auto_map = item.auto_map
    if item.use_flash_attn_2 is not None:
        config.use_flash_attn_2 = item.use_flash_attn_2
    if item.matmul_recons_thd is not None:
        config.matmul_recons_thd = item.matmul_recons_thd
    if item.fused_mlp_thd is not None:
        config.fused_mlp_thd = item.fused_mlp_thd
    if item.sdp_thd is not None:
        config.sdp_thd = item.sdp_thd
    if item.fused_attn is not None:
        config.fused_attn = item.fused_attn
    if item.matmul_fused_remap is not None:
        config.matmul_fused_remap = item.matmul_fused_remap
    if item.rmsnorm_no_half2 is not None:
        config.rmsnorm_no_half2 = item.rmsnorm_no_half2
    if item.rope_no_half2 is not None:
        config.rope_no_half2 = item.rope_no_half2
    if item.matmul_no_half2 is not None:
        config.matmul_no_half2 = item.matmul_no_half2
    if item.silu_no_half2 is not None:
        config.silu_no_half2 = item.silu_no_half2
    if item.concurrent_streams is not None:
        config.concurrent_streams = item.concurrent_streams
    global model, tokenizer, cache, generator, loaded
    model = ExLlama(config)
    tokenizer = ExLlamaTokenizer(tokenizer_path)
    cache = ExLlamaCache(model)
    generator = ExLlamaGenerator(model, tokenizer, cache) 
    loaded = True

def generation(item: GeneratorItem):
    if item.temperature is not None:
        generator.temperature = item.temperature
    if item.top_k is not None:
        generator.top_k = item.top_k
    if item.top_p is not None:
        generator.top_p = item.top_p
    if item.min_p is not None:
        generator.min_p = item.min_p
    if item.typical is not None:
        generator.typical = item.typical
    if item.token_repetition_penalty_max is not None:
        generator.token_repetition_penalty_max = item.token_repetition_penalty_max
    if item.token_repetition_penalty_sustain is not None:
        generator.token_repetition_penalty_sustain = item.token_repetition_penalty_sustain
    if item.token_repetition_penalty_decay is not None:
        generator.token_repetition_penalty_decay = item.token_repetition_penalty_decay
    if item.beams is not None:
        generator.beams = item.beams
    if item.beam_length is not None:
        generator.beam_length = item.beam_length
    if item.disallowed_tokens is not None:
        generator.disallowed_tokens = item.disallowed_tokens
    return generator.generate_simple(item.prompt, item.max_new_tokens)
    

@app.post("/load/")
async def load_model(item: LoaderItem):
    try:
        loader(item)
        return {
            "status": "success"
        }
    except Exception as e:
        print(e)
        return {
            "status": "error",
            "message": str(e)
        }
    
@app.post("/generate/")
async def generate(item: GeneratorItem):
    try:
        if not loaded:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        return generation(item)
    except Exception as e:
        print(e)
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.get("/")
async def root():
    return {
        "status": "success",
        "message": "ExLlama Risu API",
        "model": model_name,
        "loaded": loaded,
        "mode": mode
    }
