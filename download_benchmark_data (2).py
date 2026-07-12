#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchark9_V661.ipynb 基准数据集预下载脚本
=================================================
适用环境：AutoDL 国内实例（通过 hf-mirror.com 镜像下载）

功能：
  1. 下载 Benchark9_V661.ipynb 用到的 13 个 lm-eval 任务的基准数据集
     到 HuggingFace datasets 默认缓存目录（~/.cache/huggingface/datasets）
  2. 下载完成后，notebook 的 preload_datasets() 会命中缓存，离线模式可正常运行
  3. 兼容 notebook 的调用约定：
       python download_benchmark_data.py                # 下载全部
       python download_benchmark_data.py mmlu ifeval    # 只下载指定任务
       python download_benchmark_data.py --list         # 列出所有任务
       python download_benchmark_data.py --check        # 只检查缓存状态

注意：
  - 本脚本不依赖 lm_eval / vllm / torch，可在干净的新环境直接运行
  - 默认走 https://hf-mirror.com 镜像，国内直连无需代理
  - 若环境变量已设 http_proxy/https_proxy（如 source /etc/network_turbo），脚本会尊重
  - 单个任务失败不会中断整体流程，最后汇总报告
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# ============================================================
# 配置区
# ============================================================
HF_MIRROR = "https://hf-mirror.com"
DEFAULT_CACHE_DIR = os.path.expanduser("./test-data")

# 任务 → 数据集配置映射
# 每个 entry:
#   path:    HuggingFace dataset repo id
#   configs: list[str] | "all" | None(None 表示数据集无 config 概念)
#   splits:  list[str] 需要下载的 split
#   note:    说明
TASK_DATASETS = {
    "mmlu_redux_generative": {
        "path": "fxmarty/mmlu-redux-2.0-ok",
        "configs": "all",
        "splits": ["test"],
        "note": "MMLU-Redux 2.0 生成式评测（lm-eval mmlu_redux_generative 任务指定的数据集）",
    },
    "minerva_math500": {
        "path": "HuggingFaceH4/MATH-500",
        "configs": ["default"],
        "splits": ["test"],
        "note": "MATH-500（minerva 4-shot CoT 配置）",
    },
    "ifeval": {
        "path": "google/IFEval",
        "configs": None,
        "splits": ["train"],
        "note": "IFEval 指令跟随评测（仅有 train split）",
    },
    "ceval-valid": {
        "path": "ceval/ceval-exam",
        "configs": "all",
        "splits": ["val", "dev"],
        "note": "C-Eval 验证集（52 个学科，val=评测, dev=few-shot）",
    },
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "configs": ["ARC-Easy"],
        "splits": ["train", "validation", "test"],
        "note": "ARC-Easy 科学问答",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "configs": ["ARC-Challenge"],
        "splits": ["train", "validation", "test"],
        "note": "ARC-Challenge 科学问答（困难集）",
    },
    "winogrande": {
        "path": "allenai/winogrande",
        "configs": ["winogrande_xl"],
        "splits": ["train", "validation"],
        "note": "WinoGrande 共指消解（xl 配置）",
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "configs": None,
        "splits": ["train", "validation"],
        "note": "HellaSwag 常识推理（lm-eval 用 validation 评测）",
    },
    "global_piqa_completions": {
        # lm-eval 标准任务为 global_piqa_parallel/nonparallel_generation，
        # notebook 用 global_piqa_completions 可能是自定义聚合名。
        # 这里把 parallel 和 nonparallel 两个数据集都下载，覆盖所有 136 种语言。
        "path": ["mrlbenchmarks/global-piqa-parallel",
                 "mrlbenchmarks/global-piqa-nonparallel"],
        "configs": "all",
        "splits": ["test"],
        "note": "Global PIQA 多语言（136 种语言，parallel + nonparallel）",
    },
    "boolq": {
        "path": "aps/super_glue",
        "configs": ["boolq"],
        "splits": ["train", "validation"],
        "note": "BoolQ 阅读理解二分类（super_glue）",
    },
    "truthfulqa_mc2": {
        "path": "truthfulqa/truthful_qa",
        "configs": ["multiple_choice"],
        "splits": ["validation"],
        "note": "TruthfulQA MC2（多选，测幻觉）",
    },
    "humaneval": {
        "path": "openai/openai_humaneval",
        "configs": None,
        "splits": ["test"],
        "note": "HumanEval Python 代码生成 pass@1",
    },
    "bbh_cot_fewshot": {
        # lm-eval 官方已从 maveriq/bigbenchhard（loading script，datasets 5.0 不支持）
        # 迁移到 SaylorTwift/bbh（parquet 格式）。字段 input/target 与 lm-eval 的
        # doc_to_text/doc_to_target 完全匹配。
        "path": "SaylorTwift/bbh",
        "configs": [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton",
            "logical_deduction_five_objects", "logical_deduction_seven_objects",
            "logical_deduction_three_objects", "movie_recommendation",
            "multistep_arithmetic_two", "navigate", "object_counting",
            "penguins_in_a_table", "reasoning_about_colored_objects",
            "ruin_names", "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies", "word_sorting",
        ],
        "splits": ["test"],
        "note": "Big-Bench-Hard 27 子任务（3-shot CoT）",
    },
}


# ============================================================
# 环境与依赖
# ============================================================
def setup_env():
    """配置 HF 镜像、关闭 SSL 校验（autodl 自签名证书兼容）"""
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    os.environ["HF_HUB_SSL_VERIFY"] = "0"
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HF_DATASETS_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    # 不写死 autodl 代理地址（每个实例不同）；尊重已有环境变量
    # 若用户已 source /etc/network_turbo，则 http_proxy 已在 env 中
    os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR)
    # 关闭 urllib3 SSL 警告
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def ensure_deps():
    """确保 datasets / huggingface_hub / tqdm 已安装（数据下载必需的轻量依赖）"""
    missing = []
    for pkg in ["datasets", "huggingface_hub", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True
    print(f"[deps] 缺少依赖: {missing}，尝试安装...", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
             *missing],
        )
        print("[deps] 安装成功", flush=True)
        return True
    except Exception as e:
        print(f"[deps] ❌ 安装失败: {e}", flush=True)
        print(f"[deps] 请手动运行: pip install {' '.join(missing)}", flush=True)
        return False


# Benchark9_Qwen.ipynb 评测全量依赖
# (pip install 名, import 名) —— pip 名和 import 名不同的包要成对列出
# 版本锁定：vllm/lm_eval（已知可工作版本）、antlr4（lm-eval math 任务硬性要求 4.11）
BENCHMARK_DEPS = [
    ("vllm==0.25.0", "vllm"),
    ("lm_eval==0.4.12", "lm_eval"),
    ("modelscope", "modelscope"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
    ("scikit-learn", "sklearn"),
    ("langdetect", "langdetect"),
    ("tqdm", "tqdm"),
    ("httpx", "httpx"),
    ("requests", "requests"),
    ("urllib3", "urllib3"),
    ("immutabledict", "immutabledict"),          # ifeval 任务需要
    ("math_verify", "math_verify"),              # minerva_math500 任务需要
    ("antlr4-python3-runtime==4.11", "antlr4"),  # minerva_math500 硬性要求 4.11
    ("sympy", "sympy"),                          # minerva_math500 任务需要
]


def ensure_full_deps():
    """安装 Benchark9_Qwen.ipynb 全量评测依赖（vllm/lm_eval/torch 等）

    与 ensure_deps() 的区别：
      - ensure_deps() 只装数据下载必需的 3 个轻量包（datasets/huggingface_hub/tqdm）
      - ensure_full_deps() 装全部 16 个包，包括 vllm/torch（数 GB，安装慢）

    关键：装完会卸载 torchcodec（vllm 自动拉取的依赖），因为系统无 FFmpeg 时
    torchcodec 加载抛 RuntimeError（非 ImportError），绕过 vLLM 的 except ImportError
    回退，导致模型架构检查在 subprocess 中崩溃。
    """
    print("=" * 70, flush=True)
    print("[full-deps] 检查 Benchark9_Qwen.ipynb 全量评测依赖...", flush=True)
    print(f"[full-deps] 共 {len(BENCHMARK_DEPS)} 个包", flush=True)
    print("=" * 70, flush=True)

    missing_specs = []
    for pip_name, import_name in BENCHMARK_DEPS:
        try:
            __import__(import_name)
            print(f"  ✅ {import_name}", flush=True)
        except ImportError:
            missing_specs.append(pip_name)
            print(f"  ❌ {import_name} (将安装 {pip_name})", flush=True)

    if missing_specs:
        print(f"\n[full-deps] 安装 {len(missing_specs)} 个缺失包...", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
                 *missing_specs],
            )
            print("[full-deps] ✅ 安装成功", flush=True)
        except Exception as e:
            print(f"[full-deps] ❌ 安装失败: {e}", flush=True)
            print(f"[full-deps] 请手动运行: pip install {' '.join(missing_specs)}", flush=True)
            return False
    else:
        print(f"\n[full-deps] ✅ 所有依赖已就位", flush=True)

    # 关键：卸载 torchcodec（vllm 自动装的依赖，但会导致 vLLM 子进程架构检查崩溃）
    # 系统无 FFmpeg → torchcodec 加载抛 RuntimeError → vLLM 的 except ImportError 捕获不到
    # → Qwen3_5ForConditionalGeneration 架构检查失败 → 模型加载失败
    try:
        import torchcodec  # noqa: F401
        print("[full-deps] ⚠️  检测到 torchcodec，卸载中（系统无 FFmpeg，会导致 vLLM 崩溃）...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "torchcodec", "-y"],
        )
        print("[full-deps] ✅ torchcodec 已卸载", flush=True)
    except ImportError:
        print("[full-deps] ✅ torchcodec 未安装（正确状态）", flush=True)
    except Exception as e:
        print(f"[full-deps] ⚠️  torchcodec 卸载失败: {e}", flush=True)
        print("[full-deps]    请手动运行: pip uninstall torchcodec -y", flush=True)

    return True


# ============================================================
# 下载核心
# ============================================================
def _get_configs(path):
    """获取数据集的所有 config 名"""
    from datasets import get_dataset_config_names
    try:
        return get_dataset_config_names(path)
    except Exception as e:
        print(f"    ⚠️ 获取 {path} configs 失败: {e}", flush=True)
        return None


def _normalize_name(s):
    """归一化目录/路径名用于比较: 小写 + 去掉 _ 和 -
    datasets 缓存命名会把 camelCase 转 snake_case（如 IFEval → if_eval），
    还会保留 -（如 MATH-500 → math-500），用此函数统一比较。
    """
    return s.lower().replace("_", "").replace("-", "")


def _is_config_cached(path, config):
    """检查 path+config 是否已在 datasets 缓存中（粗略检查，避免重复下载）

    datasets 5.0 缓存目录命名: <namespace>___<name>
    name 可能被 snake_case 化（IFEval → if_eval），用归一化比较。
    缓存结构:
      <cache>/<namespace>___<name>/<config>/<version>/<hash>/
    无 config 时 config 目录名可能是 "default" 或 dataset name。
    """
    cache_dir = os.environ.get("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR)
    if not os.path.isdir(cache_dir):
        return False
    if "/" in path:
        namespace, name = path.split("/", 1)
    else:
        namespace, name = "", path
    # 归一化匹配 dataset 目录（datasets 会把 name snake_case 化，如 IFEval → if_eval）
    target = _normalize_name(f"{namespace}___{name}") if namespace else _normalize_name(name)
    dataset_dir = None
    for entry in os.listdir(cache_dir):
        if _normalize_name(entry) == target:
            dataset_dir = os.path.join(cache_dir, entry)
            break
    if not dataset_dir:
        return False
    if config:
        # config 目录名: 先精确匹配，再归一化兜底
        config_dir = os.path.join(dataset_dir, config)
        if os.path.isdir(config_dir) and os.listdir(config_dir):
            return True
        norm_cfg = _normalize_name(config)
        for entry in os.listdir(dataset_dir):
            if _normalize_name(entry) == norm_cfg:
                sub = os.path.join(dataset_dir, entry)
                if os.path.isdir(sub) and os.listdir(sub):
                    return True
        return False
    # 无 config: 检查任意非空子目录
    for sub in os.listdir(dataset_dir):
        sub_path = os.path.join(dataset_dir, sub)
        if os.path.isdir(sub_path) and os.listdir(sub_path):
            return True
    return False


def _download_one_config(path, config, splits):
    """下载单个 config 的指定 splits"""
    from datasets import load_dataset
    # load_dataset 会自动下载并缓存到 ~/.cache/huggingface/datasets
    # 逐 split 加载，避免一次性加载大数据集 OOM
    # datasets 5.0+ 已废弃 trust_remote_code，标准 parquet 数据集无需该参数
    loaded = {}
    for sp in splits:
        try:
            ds = load_dataset(path, name=config, split=sp)
            loaded[sp] = len(ds)
        except Exception as e:
            # 某些 split 可能不存在（如 ceval 的 dev 在部分 config 缺失），降级尝试不指定 split
            try:
                ds = load_dataset(path, name=config)
                for k in ds.keys():
                    loaded.setdefault(k, len(ds[k]))
            except Exception as e2:
                raise RuntimeError(f"split={sp} 加载失败: {e}; 不指定 split 也失败: {e2}")
    return loaded


def download_task(task_name, cfg):
    """下载单个任务对应的数据集，返回 (成功?, 详情)

    已缓存的 config 会跳过，避免重复下载。
    """
    paths = cfg["path"] if isinstance(cfg["path"], list) else [cfg["path"]]
    configs_spec = cfg["configs"]
    splits = cfg["splits"]
    details = []
    skipped = 0

    for path in paths:
        # 确定要下载的 configs
        if configs_spec is None:
            config_list = [None]
        elif configs_spec == "all":
            all_configs = _get_configs(path)
            if all_configs is None:
                return False, [f"无法获取 {path} 的 config 列表"]
            config_list = all_configs
            print(f"    {path}: 共 {len(config_list)} 个 config", flush=True)
        else:
            config_list = configs_spec

        total = len(config_list)
        for i, config in enumerate(config_list, 1):
            label = f"{path}" + (f"[{config}]" if config else "")
            # 已缓存则跳过
            if _is_config_cached(path, config):
                skipped += 1
                if total > 1:
                    print(f"    [{i}/{total}] ⏭️  已缓存，跳过 {label}", flush=True)
                else:
                    print(f"    ⏭️  已缓存，跳过 {label}", flush=True)
                continue
            if total > 1:
                print(f"    [{i}/{total}] {label}", flush=True)
            try:
                loaded = _download_one_config(path, config, splits)
                details.append(f"{label}: {loaded}")
            except Exception as e:
                # 输出格式兼容 notebook 的解析：'❌<task> 失败'
                # notebook 用 line.split('❌')[1].split(' ')[0].strip() 提取任务名，
                # 因此 ❌ 后必须紧跟任务名（无空格）
                print(f"❌{task_name} 失败 {label}: {e}", flush=True)
                return False, details + [f"失败: {label}: {e}"]
    if skipped:
        details.append(f"已缓存跳过: {skipped} 个 config")
    return True, details


# ============================================================
# 缓存检查
# ============================================================
def check_cache_status():
    """检查每个任务的数据集是否已在缓存中"""
    cache_dir = os.environ.get("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR)
    print("=" * 70)
    print("缓存状态检查")
    print(f"缓存目录: {cache_dir}")
    print("=" * 70)
    if not os.path.isdir(cache_dir):
        print(f"⚠️ 缓存目录不存在: {cache_dir}")
        return
    # datasets 库缓存目录命名: <namespace>___<name>
    # name 可能被 snake_case 化（如 IFEval → if_eval），用归一化比较
    cached = set()
    for entry in os.listdir(cache_dir):
        full = os.path.join(cache_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry in ("json", "csv", "parquet", ".locks"):
            continue
        if entry.startswith("_"):
            continue
        cached.add(_normalize_name(entry))
    print(f"已缓存的数据集 ({len(cached)}):")
    for c in sorted(cached):
        print(f"  - {c}")
    print()
    print("任务覆盖情况:")
    for task, cfg in TASK_DATASETS.items():
        paths = cfg["path"] if isinstance(cfg["path"], list) else [cfg["path"]]
        # 归一化 path 为 <namespace>___<name> 形式后比较
        norm_paths = [_normalize_name(p.replace("/", "___", 1)) for p in paths]
        hits = [p for p in norm_paths if p in cached]
        status = "✅ 已缓存" if hits else "❌ 未缓存"
        print(f"  {status}  {task:30s} <- {', '.join(paths)}")


# ============================================================
# 主流程
# ============================================================
def list_tasks():
    print("=" * 70)
    print("支持的基准任务列表")
    print("=" * 70)
    for task, cfg in TASK_DATASETS.items():
        paths = cfg["path"] if isinstance(cfg["path"], list) else [cfg["path"]]
        paths_str = " + ".join(paths)
        configs_str = (", ".join(cfg["configs"]) if isinstance(cfg["configs"], list)
                       else (cfg["configs"] or "default"))
        print(f"  {task:30s} | {paths_str}")
        print(f"  {'':30s} | configs: {configs_str} | splits: {', '.join(cfg['splits'])}")
        print(f"  {'':30s} | {cfg['note']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="下载 Benchark9_V661.ipynb 所需的基准数据集（autodl 国内环境）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_benchmark_data.py                  # 下载全部 13 个任务
  python download_benchmark_data.py mmlu_redux_generative ifeval  # 只下载指定任务
  python download_benchmark_data.py --list           # 列出所有任务
  python download_benchmark_data.py --check          # 检查缓存状态
  python download_benchmark_data.py --full-deps      # 安装全量评测依赖 + 下载数据
  python download_benchmark_data.py --full-deps --check  # 仅安装全量依赖，不下载数据
        """,
    )
    parser.add_argument("tasks", nargs="*",
                        help="要下载的任务名（不指定则下载全部）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有任务后退出")
    parser.add_argument("--check", action="store_true",
                        help="检查缓存状态后退出")
    parser.add_argument("--cache-dir", default=None,
                        help=f"HF datasets 缓存目录（默认 {DEFAULT_CACHE_DIR}）")
    parser.add_argument("--full-deps", action="store_true",
                        help="安装 Benchark9_Qwen.ipynb 全量评测依赖（vllm/lm_eval/torch 等 16 个包，"
                             "并卸载 torchcodec）。默认不启用，仅装数据下载必需的 3 个轻量包")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return 0

    setup_env()
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    # 安装全量依赖（--full-deps 时执行；放在 --check 前以支持 "--full-deps --check" 组合）
    if args.full_deps:
        if not ensure_full_deps():
            return 1

    if args.check:
        check_cache_status()
        return 0

    # 下载数据前装轻量依赖
    if not ensure_deps():
        return 1

    # 确定要下载的任务
    if args.tasks:
        unknown = [t for t in args.tasks if t not in TASK_DATASETS]
        if unknown:
            print(f"❌ 未知任务: {unknown}")
            print(f"可用任务: {list(TASK_DATASETS.keys())}")
            return 1
        tasks_to_run = args.tasks
    else:
        tasks_to_run = list(TASK_DATASETS.keys())

    print("=" * 70, flush=True)
    print(f"Benchark9_V661 基准数据集下载", flush=True)
    print(f"  镜像端点: {HF_MIRROR}", flush=True)
    print(f"  缓存目录: {os.environ.get('HF_DATASETS_CACHE', DEFAULT_CACHE_DIR)}", flush=True)
    print(f"  代理: {os.environ.get('http_proxy', '(无, 直连)')}", flush=True)
    print(f"  任务数: {len(tasks_to_run)}", flush=True)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    succeeded, failed = [], []
    t0 = time.time()
    for i, task in enumerate(tasks_to_run, 1):
        cfg = TASK_DATASETS[task]
        print(f"\n[{i}/{len(tasks_to_run)}] 📦 {task} — {cfg['note']}", flush=True)
        paths = cfg["path"] if isinstance(cfg["path"], list) else [cfg["path"]]
        print(f"    数据集: {', '.join(paths)}", flush=True)
        ts = time.time()
        ok, details = download_task(task, cfg)
        dt = time.time() - ts
        if ok:
            succeeded.append(task)
            # details 含各 config 的加载条数 + 跳过统计
            skip_info = [d for d in details if d.startswith("已缓存跳过")]
            loaded_info = [d for d in details if not d.startswith("已缓存跳过")]
            if skip_info and not loaded_info:
                print(f"    ✅ 全部已缓存 ({skip_info[0]})", flush=True)
            else:
                print(f"    ✅ 完成 ({dt:.1f}s)", flush=True)
        else:
            failed.append(task)
            print(f"    ❌ 失败 ({dt:.1f}s)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("下载汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"  ✅ 成功: {len(succeeded)}/{len(tasks_to_run)} — {succeeded}", flush=True)
    if failed:
        print(f"  ❌ 失败: {len(failed)} — {failed}", flush=True)
        print(f"\n重试失败任务:", flush=True)
        print(f"  python {' '.join(sys.argv[:1])} {' '.join(failed)}", flush=True)
    print(f"  总耗时: {time.time()-t0:.1f}s", flush=True)
    print(f"  缓存目录: {os.environ.get('HF_DATASETS_CACHE', DEFAULT_CACHE_DIR)}", flush=True)
    print("=" * 70, flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    # 兼容 notebook 调用：notebook 直接传 task 名作为位置参数
    sys.exit(main())
