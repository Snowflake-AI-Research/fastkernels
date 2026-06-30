"""Architecture registry aligned with table.tex and fastkernels baseline tasks."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Family:
    keyword: str
    display_name: str

@dataclass(frozen=True)
class Architecture:
    module: str
    family: str
    class_name: str
    model_type: str | None = None

# keyword -> full display name
_FAMILIES = (
    Family("llm", "Dense & MoE LLMs"),
    Family("linear_attn", "Linear Attention & New Archs"),
    Family("vision", "Vision / Video / Audio"),
    Family("multimodal", "Multimodal & Encoders"),
    Family("edge", "Edge & Detection"),
    Family("3d_robotics", "3D / Robotics / Science"),
    Family("recsys", "Recommendation & Specialized"),
    Family("world_models", "World Models"),
)
FAMILIES: dict[str, Family] = {f.keyword: f for f in _FAMILIES}

# L4 module stem -> Architecture
_ARCHITECTURES = (
    Architecture("llama", "llm", "Llama-3.1+", "llama"),
    Architecture("deepseek", "llm", "DeepSeek-V3.2", "deepseek_v32"),
    Architecture("mixtral", "llm", "Mixtral", "mixtral"),
    Architecture("bitnet", "llm", "BitNet 1.58b", "llama"),
    Architecture("gpt_oss", "llm", "GPT-OSS (MXFP4)", "gpt_oss"),
    Architecture("llama_eagle3", "llm", "EAGLE-3", "llama"),
    Architecture("gemma4", "llm", "Gemma-4", "gemma4"),
    Architecture("mamba", "linear_attn", "Mamba", "mamba"),
    Architecture("mamba2", "linear_attn", "Mamba2", "mamba2"),
    Architecture("rwkv7", "linear_attn", "RWKV-7", "rwkv7"),
    Architecture("gla", "linear_attn", "GLA", "gla"),
    Architecture("retnet", "linear_attn", "RetNet", "retnet"),
    Architecture("qwen3_next", "linear_attn", "Qwen-3-Next", "qwen3_next"),
    Architecture("kimi_linear", "linear_attn", "Kimi-Linear", "kimi_linear"),
    Architecture("ttt_e2e", "linear_attn", "TTT-E2E", None),
    Architecture("jamba", "linear_attn", "Jamba", "jamba"),
    Architecture("flux", "vision", "FLUX.1-Dev", None),
    Architecture("hunyuan_video", "vision", "HunyuanVideo-1.5", None),
    Architecture("sdxl", "vision", "SDXL", None),
    Architecture("sam3", "vision", "SAM3.1", "sam3_video"),
    Architecture("whisper", "vision", "Whisper", "whisper"),
    Architecture("cosyvoice3", "vision", "CosyVoice3", "cosyvoice3"),
    Architecture("qwen2_vl", "multimodal", "Qwen2-VL", "qwen2_vl"),
    Architecture("qwen3_vl", "multimodal", "Qwen3-VL", "qwen3_vl_moe"),
    Architecture("qwen2_5_omni", "multimodal", "Qwen-2.5-Omni", "qwen2_5_omni"),
    Architecture("siglip2", "multimodal", "SigLIP-2", "siglip2"),
    Architecture("dinov3", "multimodal", "DINOv3", "dinov3_vit"),
    Architecture("swinv2", "multimodal", "SwinV2", "swinv2"),
    Architecture("mobilenetv4", "edge", "MobileNetV4", None),
    Architecture("convnextv2", "edge", "ConvNeXtV2", "convnextv2"),
    Architecture("efficientnetv2", "edge", "EfficientNetV2", None),
    Architecture("yolov10", "edge", "YOLOv10", None),
    Architecture("rtdetrv2", "edge", "RTDetrV2", "rt_detr_v2"),
    Architecture("gaussian_splatting", "3d_robotics", "3DGS", None),
    Architecture("instant_ngp", "3d_robotics", "InstantNGP", None),
    Architecture("pointtransformerv3", "3d_robotics", "PointTransformerV3", None),
    Architecture("openfold3", "3d_robotics", "OpenFold3", None),
    Architecture("pi0", "3d_robotics", "Pi0", "pi0"),
    Architecture("dp3", "3d_robotics", "DP3", None),
    Architecture("dlrmv2", "recsys", "DLRMv2", None),
    Architecture("lightgcn", "recsys", "LightGCN", None),
    Architecture("bge_m3", "recsys", "BGE-M3", "xlm-roberta"),
    Architecture("colbertv2", "recsys", "ColBERTv2", "bert"),
    Architecture("llada", "recsys", "LLaDA", "llada"),
    Architecture("oasis", "world_models", "Oasis", None),
    Architecture("vjepa2", "world_models", "V-JEPA 2", "vjepa2"),
)
FASTKERNELS_ARCHITECTURES: dict[str, Architecture] = {a.module: a for a in _ARCHITECTURES}

@dataclass(frozen=True)
class BenchmarkScenario:
    module: str
    hf_name: str
    tp: int
    dtype: str
    workloads: list[str]
    num_requests: int | None = None

# Benchmarking workload registry derived from bench.yaml
DEFAULT_BENCHMARK: list[BenchmarkScenario] = [
    BenchmarkScenario("llama", "meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", ["prefill-heavy", "decode-heavy"], 100),
    BenchmarkScenario("gpt_oss", "openai/gpt-oss-120b", 2, "mxfp4", ["prefill-heavy", "decode-heavy"], 100),
    BenchmarkScenario("gla", "fla-hub/gla-2.7B-100B", 1, "bfloat16", ["prefill-heavy", "decode-heavy"], 100),
    BenchmarkScenario("flux", "black-forest-labs/FLUX.1-dev", 1, "bfloat16", ["1024x1024", "512x512"]),
    BenchmarkScenario("qwen3_vl", "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 4, "fp8", ["text-only", "image", "video"], 100),
    BenchmarkScenario("yolov10", "jameslahm/yolov10n", 1, "bfloat16", ["coco-val"]),
    BenchmarkScenario("openfold3", "OpenFold/OpenFold3", 1, "bfloat16", ["short", "medium", "long", "extra-long"]),
    BenchmarkScenario("bge_m3", "BAAI/bge-m3", 1, "bfloat16", ["bge-m3-mldr-docs"]),
    BenchmarkScenario("oasis", "Etched/oasis-500m", 1, "float16", ["short-bs4-16f-4ddim", "medium-bs8-24f-4ddim", "long-bs8-32f-4ddim", "denoise-bs4-16f-8ddim"]),
]

def print_benchmark_set(benchmarks: list[BenchmarkScenario], title: str = "BENCHMARK SCENARIOS") -> None:
    """Print a benchmark set in a tabular format with line-wrapping for long columns."""
    import textwrap
    
    header = f"{'Module':<12} | {'Family':<12} | {'HF Name':<38} | {'TP':<2} | {'dtype':<8} | {'#req':<4} | {'Workloads'}"
    width = 118  # Fixed width that easily fits in standard terminals
    
    print("\n")
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print(header)
    print("-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 38 + "-+-" + "-" * 2 + "-+-" + "-" * 8 + "-+-" + "-" * 4 + "-+-" + "-" * 23)
    
    for bs in benchmarks:
        arch = FASTKERNELS_ARCHITECTURES.get(bs.module)
        family = arch.family if arch else "Unknown"
        num_req = str(bs.num_requests) if bs.num_requests is not None else "-"
        wloads = ", ".join(bs.workloads)
        
        # Wrap the workloads column so it doesn't blow out the terminal width
        wrapped_wloads = textwrap.wrap(wloads, width=25, break_on_hyphens=False)
        if not wrapped_wloads:
            wrapped_wloads = [""]
            
        for i, line in enumerate(wrapped_wloads):
            if i == 0:
                print(f"{bs.module:<12} | {family:<12} | {bs.hf_name:<38} | {bs.tp:<2} | {bs.dtype:<8} | {num_req:<4} | {line}")
            else:
                print(f"{'':<12} | {'':<12} | {'':<38} | {'':<2} | {'':<8} | {'':<4} | {line}")
    print("=" * width)

def print_registry() -> None:
    """Print the families and FASTKERNELS_ARCHITECTURES in a tabular format."""
    fam_header = f"{'Family Name':<30} | {'Keyword'}"
    fam_rows = [f"{fam.display_name:<30} | {fam.keyword}" for fam in FAMILIES.values()]
    fam_width = max(len(fam_header), max((len(r) for r in fam_rows), default=0))
    
    print("=" * fam_width)
    print(f"{'FAMILIES':^{fam_width}}")
    print("=" * fam_width)
    print(fam_header)
    print("-" * 30 + "-+-" + "-" * (fam_width - 33))
    for row in fam_rows:
        print(row)
    print("\n")
    
    arch_header = f"{'Architecture':<20} | {'L4 Module':<20} | {'HuggingFace model_type':<24} | {'Family'}"
    arch_rows = []
    for arch in FASTKERNELS_ARCHITECTURES.values():
        m_type = str(arch.model_type) if arch.model_type is not None else "None"
        arch_rows.append(f"{arch.class_name:<20} | {arch.module:<20} | {m_type:<24} | {arch.family}")
        
    arch_width = max(len(arch_header), max((len(r) for r in arch_rows), default=0))
    
    print("=" * arch_width)
    print(f"{'FASTKERNELS ARCHITECTURES':^{arch_width}}")
    print("=" * arch_width)
    print(arch_header)
    print("-" * 20 + "-+-" + "-" * 20 + "-+-" + "-" * 24 + "-+-" + "-" * (arch_width - 73))
    for row in arch_rows:
        print(row)
    print("=" * arch_width)
    
    print_benchmark_set(DEFAULT_BENCHMARK, "DEFAULT BENCHMARK")

if __name__ == "__main__":
    print_registry()