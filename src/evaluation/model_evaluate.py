# -*- coding: utf-8 -*-
"""
纯 LLM 评估脚本 (多卡并行版 v9.2)
- 基础功能: V9 (ID锚定聚合 + SFT适配 + 分片评估)
- 支持两类“训练后权重”加载方式：
  A) LoRA/PEFT adapter 目录（你现在这种：adapter_config.json + adapter_model.safetensors）
     用法：--adapter_dir /path/to/ckpt-stepXX-updYY
  B) 单文件 .pt / state_dict（兼容旧逻辑）
     用法：--checkpoint_pt /path/to/ckpt.pt

"""

from __future__ import annotations

import os
import re
import json
import argparse
import time
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta

from tqdm import tqdm
try:
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    # Allow `python evaluate.py --help` on machines without torch/transformers.
    torch = None  # type: ignore
    dist = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

# ====== 新增：PEFT/LoRA 支持 ======
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# ===========================
# 1. 基础工具 & 正则
# ===========================
ROOT_FALLBACK_KEY = "__ROOT__"
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MODEL_PREFIX_RE = re.compile(r"^\s*[\w\-]+:\s*", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)


def extract_username_from_sft_format(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _USERNAME_LINE_RE.search(text)
    if m:
        return m.group("name").strip()
    return ""


def extract_pure_content_from_sft_format(text: str) -> str:
    if not isinstance(text, str):
        return ""
    start_marker = "content:\n"
    idx_start = text.find(start_marker)
    if idx_start == -1:
        start_marker = "content:"
        idx_start = text.find(start_marker)
    if idx_start == -1:
        return text

    idx_end = text.find("\nuserinterest:", idx_start)
    if idx_end == -1:
        idx_end = text.find("userinterest:", idx_start)

    if idx_end != -1:
        pure_content = text[idx_start + len(start_marker):idx_end].strip()
    else:
        idx_end_pot = text.find("\npotentials:", idx_start)
        if idx_end_pot != -1:
            pure_content = text[idx_start + len(start_marker):idx_end_pot].strip()
        else:
            pure_content = text[idx_start + len(start_marker):].strip()
    return pure_content


def extract_potentialspan_from_text(user_content_str: str) -> List[Dict[str, Any]]:
    if not isinstance(user_content_str, str):
        return []
    for prefix in ["potentials:", "potentialspan:"]:
        idx = user_content_str.find(prefix)
        if idx >= 0:
            span_text = user_content_str[idx + len(prefix):].strip()
            try:
                arr = json.loads(span_text)
                if isinstance(arr, list):
                    return arr
            except Exception:
                pass
    return []


def render_potentialspan_json(pots: List[Dict[str, Any]], depth: int) -> str:
    dval = int(depth)
    blocks = []
    for p in (pots or []):
        blk = {
            "user_name": (p.get("user_name") or "").strip(),
            "interests": p.get("interests") or [],
            "depth": dval,
            "interaction_count": int(p.get("interaction_count", 0)),
        }
        if blk["user_name"]:
            blocks.append(blk)
    return json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))


def strip_model_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = _MODEL_PREFIX_RE.sub("", s, count=1)
    m = _CODE_FENCE_RE.search(s)
    if m:
        s = m.group("body").strip()
    return s


def _infer_max_input_tokens(tokenizer, model, default: int = 8192) -> int:
    """
    Infer a safe max input length for the current model/tokenizer.
    - Some tokenizers expose a huge `model_max_length` sentinel; filter it out.
    - Also consult model.config.* if available.
    """
    cand = []
    try:
        ml = int(getattr(tokenizer, "model_max_length", 0) or 0)
        if 0 < ml < 100000:
            cand.append(ml)
    except Exception:
        pass
    try:
        c = getattr(model, "config", None)
        if c is not None:
            for k in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
                v = getattr(c, k, None)
                if v is None:
                    continue
                try:
                    v = int(v)
                except Exception:
                    continue
                if v and v > 0:
                    cand.append(v)
    except Exception:
        pass
    if not cand:
        return int(default)
    return int(min(cand + [int(default)]))


def parse_model_output(gen_text: str, cand_order: List[str]) -> tuple[List[Dict[str, Any]], bool]:
    clean_text = strip_model_output(gen_text)
    l = clean_text.find("[")
    r = clean_text.rfind("]")
    json_str = clean_text[l:r + 1] if (l != -1 and r != -1 and r > l) else clean_text

    try:
        data = json.loads(json_str)
    except Exception:
        return [], False

    if not isinstance(data, list):
        return [], False

    data_map = {}
    for item in data:
        if isinstance(item, dict):
            u = item.get("user_name")
            if u:
                data_map[u] = item

    results = []
    for i, uname in enumerate(cand_order):
        pred_item = None
        if i < len(data) and isinstance(data[i], dict):
            if data[i].get("user_name") == uname or not data[i].get("user_name"):
                pred_item = data[i]
        if pred_item is None:
            pred_item = data_map.get(uname)

        res = {"user_name": uname, "type": 0, "content": ""}
        if pred_item:
            try:
                t = int(pred_item.get("type", 0))
            except Exception:
                t = 0
            res["type"] = t
            res["content"] = str(pred_item.get("content", "")).strip()
        results.append(res)
    return results, True


def smart_extract_potentials(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rm = json_data.get("reward_model", {})
    rp = rm.get("root_potential", {})
    if rp.get("full") and isinstance(rp["full"], list):
        return rp["full"]
    raw_content = json_data.get("content", "")
    pots_from_text = extract_potentialspan_from_text(raw_content)
    if pots_from_text:
        return pots_from_text
    if rp.get("user_names") and isinstance(rp["user_names"], list):
        return [{"user_name": n, "interests": [], "interaction_count": 0} for n in rp["user_names"] if n]
    return []


# ===========================
# 2. 全局池管理
# ===========================
class GlobalPoolManager:
    def __init__(self):
        self.pools: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def preload(self, rows: List[Dict[str, Any]], rank: int):
        iterator = tqdm(rows, desc=f"Rank {rank} Preloading", disable=(rank != 0))
        for row in iterator:
            sft_info = row.get("sft_chunk_info") or {}
            record_id = sft_info.get("record_id")
            if not record_id:
                continue

            user_msgs = [m for m in row.get("prompt", []) if m.get("role") == "user"]
            if not user_msgs:
                continue
            content_str = user_msgs[0].get("content", "")

            pots = []
            try:
                data = json.loads(content_str)
                if isinstance(data, dict):
                    pots = smart_extract_potentials(data)
            except Exception:
                pass
            if not pots:
                pots = extract_potentialspan_from_text(content_str)

            for p in pots:
                uname = p.get("user_name")
                if uname:
                    old_p = self.pools[record_id].get(uname)
                    if (not old_p) or (not old_p.get("interests") and p.get("interests")):
                        self.pools[record_id][uname] = p

        if rank == 0:
            valid_trees = len(self.pools)
            total_cands = sum(len(v) for v in self.pools.values())
            print(f"[GlobalPool] Aggregated {valid_trees} trees. Total candidates: {total_cands}")

    def get_candidates(self, record_id: str) -> List[Dict[str, Any]]:
        return list(self.pools.get(record_id, {}).values())

    def find_interests(self, record_id: str, username: str) -> List[str]:
        u = self.pools.get(record_id, {}).get(username)
        return u.get("interests") or [] if u else []


# ===========================
# 3. 推理逻辑
# ===========================
def step_rollout(
    model,
    tokenizer,
    device,
    system_prompt: str,
    base_user_content: str,
    candidates: List[Dict[str, Any]],
    depth: int,
    gen_params: Dict[str, Any],
):
    if torch is None:
        raise RuntimeError("torch 未安装，无法运行评估。请在含 torch/transformers 的环境运行 evaluate.py。")
    pot_json = render_potentialspan_json(candidates, depth)
    clean_base = base_user_content.split("\npotentials:")[0].split("\npotentialspan:")[0].strip()
    if clean_base.startswith("{") and clean_base.endswith("}"):
        try:
            js = json.loads(clean_base)
            if "content" in js:
                clean_base = js["content"]
        except Exception:
            pass

    final_content = clean_base + "\npotentials: " + pot_json
    cand_order = [p["user_name"] for p in candidates]

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": final_content}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    max_inp = int(gen_params.get("max_input_tokens", 8192) or 8192)
    # Truncate at tokenization time to avoid >model_max_length warnings and position index errors.
    inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=max_inp).to(device)

    if inputs["input_ids"].shape[1] > max_inp:
        inputs["input_ids"] = inputs["input_ids"][:, -max_inp:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_inp:]

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                max_new_tokens=gen_params.get("max_new_tokens", 4096),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = out[0, inputs["input_ids"].shape[1]:]
            raw_gen = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except Exception:
        raw_gen = "[]"

    if not cand_order:
        return [], chat_text, raw_gen
    preds, _ = parse_model_output(raw_gen, cand_order)
    return preds, chat_text, raw_gen


# ===========================
# 4. 分布式初始化
# ===========================
def setup_distributed():
    if torch is None or dist is None:
        return 0, 1, 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 增加超时时间，避免长尾导致 watchdog 误杀（评估一般不会很夸张，这里给 5h）
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=5))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _safe_read_json_lines(path: str) -> list[tuple[str, Optional[dict]]]:
    """
    Read a JSONL file and parse each line.
    - Returns a list of (raw_line, parsed_obj_or_None).
    - If a line fails to parse, parsed obj is None (we treat it as a partial/corrupted tail candidate).
    """
    out: list[tuple[str, Optional[dict]]] = []
    if not path or (not os.path.exists(path)):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
            out.append((raw, obj))
    return out

def _truncate_jsonl_to_first_n_record_done(path: str, keep_done_count: int) -> int:
    """
    Truncate JSONL to keep the first N completed records (as marked by {"_event":"record_done"}).
    Also drops any trailing unparsable lines.
    Returns the number of record_done markers kept (should equal keep_done_count if enough).
    """
    if not path or (not os.path.exists(path)):
        return 0
    pairs = _safe_read_json_lines(path)
    if not pairs:
        return 0
    done = 0
    cut_at_raw_line_idx_exclusive = 0
    for i, (_raw, obj) in enumerate(pairs):
        if obj is None:
            # Unparsable: treat as tail corruption; stop here.
            break
        cut_at_raw_line_idx_exclusive = i + 1
        if isinstance(obj, dict) and obj.get("_event") == "record_done":
            done += 1
            if done >= keep_done_count:
                # Keep through this marker.
                break
    # If we didn't reach keep_done_count, we still truncate to the last valid JSON line.
    # If we did reach it, we truncate to include the record_done marker.
    with open(path, "w", encoding="utf-8") as f:
        for j in range(cut_at_raw_line_idx_exclusive):
            raw, obj = pairs[j]
            if obj is None:
                break
            # Ensure newline
            if raw.endswith("\n"):
                f.write(raw)
            else:
                f.write(raw + "\n")
    # Recount kept markers (robust)
    kept = 0
    for j in range(cut_at_raw_line_idx_exclusive):
        _raw, obj = pairs[j]
        if isinstance(obj, dict) and obj.get("_event") == "record_done":
            kept += 1
    return kept

def _count_record_done(path: str) -> int:
    if not path or (not os.path.exists(path)):
        return 0
    pairs = _safe_read_json_lines(path)
    n = 0
    for _raw, obj in pairs:
        if isinstance(obj, dict) and obj.get("_event") == "record_done":
            n += 1
    return n

def _count_legacy_completed_records_drop_last_group(path: str) -> int:
    """
    Legacy (pre-record_done) resume:
    The old log format wrote per-node JSON lines and had no explicit record boundary marker.
    We infer boundaries by contiguous blocks of the same group_id:
      group_id=A (many lines), group_id=B (many lines), ...
    If the process was interrupted mid-record, the last group's block is incomplete.
    To satisfy the user's "完全续评" semantics, we DROP the last group block.
    Returns: number of completed record blocks to keep (>=0).
    """
    if not path or (not os.path.exists(path)):
        return 0
    pairs = _safe_read_json_lines(path)
    last_gid = None
    blocks = 0
    for _raw, obj in pairs:
        if not isinstance(obj, dict):
            # Legacy logs *should* be valid JSON per line, but in practice a single corrupted line
            # can exist (e.g., partial write). We skip such lines instead of stopping early,
            # otherwise resume may severely undercount (e.g., 287 -> 106).
            continue
        gid = obj.get("group_id")
        if not gid:
            # ignore non-node lines if any
            continue
        if last_gid is None:
            last_gid = gid
            blocks = 1
        elif gid != last_gid:
            blocks += 1
            last_gid = gid
    # drop the last group as "possibly incomplete"
    return max(0, blocks - 1)

def _file_exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False

def _peek_last_group_id(path: str) -> str:
    """Best-effort: return last parsed group_id in a JSONL, else empty string."""
    if not path or (not os.path.exists(path)):
        return ""
    pairs = _safe_read_json_lines(path)
    last_gid = ""
    for _raw, obj in pairs:
        if isinstance(obj, dict) and obj.get("group_id"):
            last_gid = str(obj.get("group_id"))
    return last_gid

def _peek_last_event_seq(path: str) -> Tuple[str, int]:
    """Best-effort: return (last_event, last_seq) for new logs, else ("", -1)."""
    if not path or (not os.path.exists(path)):
        return "", -1
    pairs = _safe_read_json_lines(path)
    last_event = ""
    last_seq = -1
    for _raw, obj in pairs:
        if not isinstance(obj, dict):
            continue
        ev = obj.get("_event")
        if ev:
            last_event = str(ev)
        if "_seq" in obj:
            try:
                last_seq = int(obj.get("_seq"))
            except Exception:
                pass
    return last_event, last_seq

def _iter_legacy_group_blocks(path: str) -> list[str]:
    """
    Legacy logs: infer record boundaries by contiguous group_id blocks.
    Returns a list of group_id in block order.
    Skips corrupted (unparsable) lines.
    """
    blocks: list[str] = []
    if not path or (not os.path.exists(path)):
        return blocks
    pairs = _safe_read_json_lines(path)
    last_gid = None
    for _raw, obj in pairs:
        if not isinstance(obj, dict):
            continue
        gid = obj.get("group_id")
        if not gid:
            continue
        gid = str(gid)
        if last_gid is None or gid != last_gid:
            blocks.append(gid)
            last_gid = gid
    return blocks

def _find_seq_for_group_id_in_indices(group_id: str, my_indices: list[int], rows: list[dict]) -> Optional[int]:
    """
    Map a legacy log group_id back to this rank's seq (index in my_indices),
    using the same record_id logic as the main loop: record_id = sft_chunk_info.record_id or f"row{ridx}".
    Returns seq if found, else None.
    """
    target = (group_id or "").strip()
    if not target:
        return None
    for seq, ridx in enumerate(my_indices):
        try:
            row = rows[ridx]
        except Exception:
            continue
        sft_info = row.get("sft_chunk_info") or {}
        record_id = sft_info.get("record_id") or f"row{ridx}"
        if str(record_id) == target:
            return seq
    return None

def _truncate_jsonl_to_last_record_done(path: str) -> Optional[int]:
    """
    For new logs (with record_done markers):
    - Truncate file to keep everything up to and including the last record_done line.
    - Returns next_seq (= last_record_done._seq + 1) if available, else (#record_done kept).
    - Returns None if no record_done marker is found.
    """
    if not path or (not os.path.exists(path)):
        return None
    pairs = _safe_read_json_lines(path)
    if not pairs:
        return None
    last_done_idx = None
    last_done_seq = None
    done_count = 0
    for i, (_raw, obj) in enumerate(pairs):
        if not isinstance(obj, dict):
            continue
        if obj.get("_event") == "record_done":
            last_done_idx = i
            done_count += 1
            if "_seq" in obj:
                try:
                    last_done_seq = int(obj.get("_seq"))
                except Exception:
                    pass
    if last_done_idx is None:
        return None

    with open(path, "w", encoding="utf-8") as f:
        for j in range(last_done_idx + 1):
            raw, obj = pairs[j]
            if not isinstance(obj, dict):
                # drop corrupted lines
                continue
            if raw.endswith("\n"):
                f.write(raw)
            else:
                f.write(raw + "\n")

    if last_done_seq is not None:
        return last_done_seq + 1
    return done_count

def _truncate_jsonl_to_first_n_legacy_groups(path: str, keep_groups: int) -> int:
    """
    Truncate a legacy JSONL file to keep the first N group_id blocks (contiguous groups),
    dropping everything from the start of group (N+1) onward.
    Also drops any trailing unparsable lines.
    Returns the number of groups kept (<=keep_groups if file shorter).
    """
    if not path or (not os.path.exists(path)):
        return 0
    pairs = _safe_read_json_lines(path)
    if not pairs:
        return 0
    if keep_groups <= 0:
        # clear file
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return 0

    last_gid = None
    blocks = 0
    cut_at_raw_line_idx_exclusive = 0
    for i, (raw, obj) in enumerate(pairs):
        if not isinstance(obj, dict):
            # Skip corrupted lines; do not stop counting/keeping.
            continue
        gid = obj.get("group_id")
        if not gid:
            # keep non-group lines only if we're still within kept region
            cut_at_raw_line_idx_exclusive = i + 1
            continue

        if last_gid is None:
            last_gid = gid
            blocks = 1
        elif gid != last_gid:
            # new group block begins
            blocks += 1
            last_gid = gid
        if blocks > keep_groups:
            # stop before first line of (keep_groups+1)th block
            break
        cut_at_raw_line_idx_exclusive = i + 1

    with open(path, "w", encoding="utf-8") as f:
        for j in range(cut_at_raw_line_idx_exclusive):
            raw, obj = pairs[j]
            if not isinstance(obj, dict):
                # Drop corrupted lines during truncation.
                continue
            if raw.endswith("\n"):
                f.write(raw)
            else:
                f.write(raw + "\n")
    return min(blocks, keep_groups)

def _sync_resume_and_truncate(
    *,
    jsonl_detail_prefix: str,
    jsonl_io_prefix: Optional[str],
    world_size: int,
    rank: int,
    device: torch.device,
    resume: bool,
) -> int:
    """
    Multi-GPU safe resume:
    - rank0 inspects all rank files, computes global_min_done (min completed records across ranks),
      and truncates every rank file to keep exactly those completed records.
    - broadcasts start_seq (=global_min_done) to all ranks.
    Returns start_seq for this rank.
    """
    start_seq = 0
    if not resume:
        return 0

    if world_size <= 1 or (not (dist.is_available() and dist.is_initialized())):
        # single GPU / no distributed: local resume
        det_path = f"{jsonl_detail_prefix}.rank{rank}"
        io_path = f"{jsonl_io_prefix}.rank{rank}" if jsonl_io_prefix else None
        local_done = _count_record_done(det_path)
        if local_done > 0:
            _truncate_jsonl_to_first_n_record_done(det_path, local_done)
            if io_path:
                _truncate_jsonl_to_first_n_record_done(io_path, local_done)
            return local_done
        # legacy fallback: drop last group_id block
        legacy_keep = _count_legacy_completed_records_drop_last_group(det_path)
        _truncate_jsonl_to_first_n_legacy_groups(det_path, legacy_keep)
        if io_path:
            _truncate_jsonl_to_first_n_legacy_groups(io_path, legacy_keep)
        return legacy_keep

    # distributed
    if rank == 0:
        done_counts = []
        has_markers = False
        det_paths = []
        for r in range(world_size):
            det_path = f"{jsonl_detail_prefix}.rank{r}"
            det_paths.append(det_path)
            c = _count_record_done(det_path)
            if c > 0:
                has_markers = True
            done_counts.append(c)

        if has_markers:
            global_min_done = int(min(done_counts)) if done_counts else 0
            # helpful diagnostics
            msg_parts = []
            for r, det_path in enumerate(det_paths):
                ok = _file_exists(det_path)
                ev, last_seq = _peek_last_event_seq(det_path) if ok else ("", -1)
                msg_parts.append(f"rank{r}:exists={ok},done={done_counts[r]},tail=({ev},{last_seq})")
            print("[Resume] marker-mode counts: " + " | ".join(msg_parts), flush=True)
            print(f"[Resume] marker-mode global_min_done={global_min_done}", flush=True)
            # Truncate all ranks to global_min_done by record_done markers
            for r in range(world_size):
                det_path = f"{jsonl_detail_prefix}.rank{r}"
                _truncate_jsonl_to_first_n_record_done(det_path, global_min_done)
                if jsonl_io_prefix:
                    io_path = f"{jsonl_io_prefix}.rank{r}"
                    _truncate_jsonl_to_first_n_record_done(io_path, global_min_done)
            start_seq = global_min_done
        else:
            # Legacy fallback: infer groups and drop last group on each rank, then align by min.
            legacy_counts = []
            for r in range(world_size):
                det_path = f"{jsonl_detail_prefix}.rank{r}"
                legacy_counts.append(_count_legacy_completed_records_drop_last_group(det_path))
            global_min_keep = int(min(legacy_counts)) if legacy_counts else 0
            # helpful diagnostics
            msg_parts = []
            for r in range(world_size):
                det_path = f"{jsonl_detail_prefix}.rank{r}"
                ok = _file_exists(det_path)
                last_gid = _peek_last_group_id(det_path) if ok else ""
                msg_parts.append(f"rank{r}:exists={ok},keep={legacy_counts[r]},last_group_id={last_gid}")
            print("[Resume] legacy-mode inferred counts (drop-last-block): " + " | ".join(msg_parts), flush=True)
            print(f"[Resume] legacy-mode global_min_keep={global_min_keep}", flush=True)
            for r in range(world_size):
                det_path = f"{jsonl_detail_prefix}.rank{r}"
                _truncate_jsonl_to_first_n_legacy_groups(det_path, global_min_keep)
                if jsonl_io_prefix:
                    io_path = f"{jsonl_io_prefix}.rank{r}"
                    _truncate_jsonl_to_first_n_legacy_groups(io_path, global_min_keep)
            start_seq = global_min_keep

    # broadcast start_seq
    t = torch.tensor([start_seq], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


# ===========================
# 5. 核心：加载 base +（可选）LoRA adapter 或 .pt
# ===========================
def _is_adapter_dir(p: str) -> bool:
    if not p or (not os.path.isdir(p)):
        return False
    return os.path.exists(os.path.join(p, "adapter_config.json"))


def load_model_with_optional_adapters(
    *,
    model_path: str,
    tokenizer_path: str,
    local_rank: int,
    adapter_dir: str = None,
    checkpoint_pt: str = None,
    merge_lora: bool = False,
    rank: int = 0,
):
    """
    优先级：
      1) adapter_dir（目录且包含 adapter_config.json） -> LoRA
      2) checkpoint_pt 如果是 adapter_dir -> LoRA
      3) checkpoint_pt 如果是 .pt 文件 -> load_state_dict(strict=False)
      4) 都没有 -> base
    """
    if tokenizer_path is None:
        tokenizer_path = model_path

    # base
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": local_rank},
        torch_dtype="auto",
        trust_remote_code=True,
    )
    base.eval()

    # decide adapter path
    adapter_path = None
    if adapter_dir and _is_adapter_dir(adapter_dir):
        adapter_path = adapter_dir
    elif checkpoint_pt and _is_adapter_dir(checkpoint_pt):
        adapter_path = checkpoint_pt

    # case A: LoRA adapter dir
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("未安装 peft，但你提供了 LoRA adapter_dir。请先 pip install peft")
        if rank == 0:
            print(f"[Init] Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        model.eval()
        if merge_lora:
            # 合并后推理更快，但会占用额外显存/时间；显存紧就别开
            if rank == 0:
                print("[Init] Merging LoRA into base weights (merge_and_unload=True)")
            model = model.merge_and_unload()
            model.eval()
        return model

    # case B: .pt state_dict
    if checkpoint_pt and os.path.isfile(checkpoint_pt) and checkpoint_pt.endswith(".pt"):
        if rank == 0:
            print(f"[Init] Loading Checkpoint from .pt file: {checkpoint_pt}")
        state_dict = torch.load(checkpoint_pt, map_location="cpu")
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("_orig_mod."):
                k = k[10:]
            new_state_dict[k] = v

        load_result = base.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            print(f"[Init] Weights Loaded (.pt). Missing keys: {len(load_result.missing_keys)}, "
                  f"Unexpected keys: {len(load_result.unexpected_keys)}")
            if load_result.unexpected_keys:
                print(f"[Init] Sample unexpected keys: {load_result.unexpected_keys[:3]}")
        base.eval()
        return base

    # base only
    if rank == 0:
        if adapter_dir or checkpoint_pt:
            print("[WARN] Provided adapter_dir/checkpoint_pt but none is usable. Falling back to base only.")
    return base


# ===========================
# 6. 主评估流程
# ===========================
def evaluate_jsonl(
    data_path,
    model_path,
    checkpoint_pt,
    tokenizer_path,
    jsonl_detail,
    jsonl_io,
    max_samples=10000,
    max_turns=0,
    depth_limit=1,
    adapter_dir=None,
    merge_lora=False,
    group_by_record_id: bool = False,
    root_chunk_size: int = 0,
    candidate_chunk_size: int = 50,
    resume: bool = False,
):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("缺少 torch/transformers，无法运行评估。请在含 torch/transformers 的环境运行 evaluate.py。")
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[Init] World Size: {world_size}, Base Model: {model_path}")
        if adapter_dir:
            print(f"[Init] adapter_dir: {adapter_dir}")
        if checkpoint_pt:
            print(f"[Init] checkpoint_pt: {checkpoint_pt}")

    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = load_model_with_optional_adapters(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        local_rank=local_rank,
        adapter_dir=adapter_dir,
        checkpoint_pt=checkpoint_pt,
        merge_lora=merge_lora,
        rank=rank,
    )
    model.eval()

    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if rank == 0:
        print(f"[Data] Loaded {len(rows)} raw lines.")

    pool_mgr = GlobalPoolManager()
    pool_mgr.preload(rows, rank)

    detail_file_rank = f"{jsonl_detail}.rank{rank}"
    io_file_rank = f"{jsonl_io}.rank{rank}" if jsonl_io else None
    ensure_dir(detail_file_rank)

    # =========
    # NEW: group-by record_id mode (多行候选池分片只评一次)
    # =========
    rep_by_record_id: Dict[str, Dict[str, Any]] = {}
    record_order: List[str] = []
    if group_by_record_id:
        for ridx, row in enumerate(rows):
            sft_info = row.get("sft_chunk_info") or {}
            record_id = sft_info.get("record_id") or f"row{ridx}"
            record_id = str(record_id)
            if record_id not in rep_by_record_id:
                rep_by_record_id[record_id] = row
                record_order.append(record_id)
        if rank == 0:
            print(f"[Data] group_by_record_id=True => unique record_id={len(record_order)}", flush=True)

        # distribute record_ids across ranks
        my_record_ids = record_order[rank::world_size]
        my_max_samples = max_samples // world_size
        my_limit = min(len(my_record_ids), my_max_samples)
    else:
        my_max_samples = max_samples // world_size
        my_indices = list(range(rank, len(rows), world_size))
        my_limit = min(len(my_indices), my_max_samples)

    # Resume support (IMPORTANT):
    # - Evaluation is data-parallel across ranks.
    # - group_by_record_id: ranks process disjoint record_id lists
    # - legacy mode: ranks process disjoint row indices
    start_seq = 0
    resume_mode = "none"
    if resume:
        # 1) New logs: use record_done markers (truncate to last marker; resume from marker._seq + 1)
        next_seq = _truncate_jsonl_to_last_record_done(detail_file_rank)
        if next_seq is not None:
            start_seq = int(next_seq)
            resume_mode = "marker"
            if io_file_rank and os.path.exists(io_file_rank):
                _truncate_jsonl_to_last_record_done(io_file_rank)
        else:
            # 2) Legacy logs: infer group_id blocks; DROP the last block; truncate to kept blocks;
            #    then map last kept group_id back to seq in my_indices.
            blocks = _iter_legacy_group_blocks(detail_file_rank)
            keep_blocks = max(0, len(blocks) - 1)  # drop last (possibly incomplete)
            if keep_blocks > 0:
                last_complete_gid = blocks[keep_blocks - 1]
                _truncate_jsonl_to_first_n_legacy_groups(detail_file_rank, keep_blocks)
                if io_file_rank and os.path.exists(io_file_rank):
                    _truncate_jsonl_to_first_n_legacy_groups(io_file_rank, keep_blocks)
                seq_last = _find_seq_for_group_id_in_indices(last_complete_gid, my_indices, rows)
                if seq_last is not None:
                    start_seq = int(seq_last) + 1
                else:
                    # Fallback: best-effort (may be conservative if some indices were skipped and not logged)
                    start_seq = keep_blocks
                resume_mode = "legacy"
            else:
                # Nothing reliable to resume from
                start_seq = 0
                resume_mode = "legacy_empty"

        print(f"[Resume] rank={rank} mode={resume_mode} start_seq={start_seq}", flush=True)

    f_det = open(detail_file_rank, "a" if resume else "w", encoding="utf-8")
    f_io = open(io_file_rank, "a" if resume else "w", encoding="utf-8") if io_file_rank else None

    safe_max_inp = _infer_max_input_tokens(tokenizer, model, default=8192)
    gen_params = {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 4096, "max_input_tokens": safe_max_inp}
    if rank == 0:
        print(f"[Init] max_input_tokens set to {safe_max_inp} (clamped by model/tokenizer)", flush=True)
    DYNAMIC_CHUNK_SIZE = int(candidate_chunk_size or 50)
    if DYNAMIC_CHUNK_SIZE <= 0:
        DYNAMIC_CHUNK_SIZE = 50

    # We limit by per-rank sample ordinal (seq), not by "processed in this run",
    # otherwise resume would re-run up to my_max_samples again.
    start_at = int(start_seq) if resume else 0
    start_at = max(0, min(start_at, my_limit))

    iterator = tqdm(
        total=my_limit,
        initial=start_at,
        desc=f"Rank {rank} Eval",
        disable=(rank != 0),
    )

    processed_since_resume = 0
    for seq in range(start_at, my_limit):
        if rank == 0:
            iterator.update(1)

        if group_by_record_id:
            record_id = my_record_ids[seq]
            row = rep_by_record_id.get(record_id) or {}
            ridx = -1
        else:
            ridx = my_indices[seq]
            row = rows[ridx]
            sft_info = row.get("sft_chunk_info") or {}
            record_id = sft_info.get("record_id") or f"row{ridx}"
            record_id = str(record_id)

        prompt = row.get("prompt") or []
        sft_info = row.get("sft_chunk_info") or {}

        sys_prompt = ""
        root_user_json = None
        for m in prompt:
            if m.get("role") == "system":
                sys_prompt = m.get("content", "")
            if m.get("role") == "user" and root_user_json is None:
                try:
                    root_user_json = json.loads(m.get("content", ""))
                except Exception:
                    root_user_json = {"content": m.get("content", "")}

        if not root_user_json:
            continue

        raw_content_str = root_user_json.get("content", "")
        pure_root_content = extract_pure_content_from_sft_format(raw_content_str)

        root_user_json["depth"] = 0
        root_user_json["_step_depth"] = 0
        if not root_user_json.get("user_name"):
            root_user_json["user_name"] = extract_username_from_sft_format(raw_content_str)

        queue = deque([root_user_json])
        root_username = root_user_json.get("user_name", "")

        # max_turns: 每个 record 最多处理的节点数（BFS 出队次数）
        # - max_turns <= 0: 不限制，直到队列为空（仍受 depth_limit 约束）
        # - max_turns > 0 : 限制最多处理 N 个节点
        node_limit = int(max_turns or 0)
        steps_done = 0
        # 仅统计规模（不依赖GT）：L1/L2 预测为正(type in 1/2)的人数，以及其中评论(type==1)人数
        tree_stats = {
            "l1_pos_total": 0,
            "l1_pos_comment": 0,
            "l1_pos_repost": 0,
            "l2_pos_total": 0,
            "l2_pos_comment": 0,
            "l2_pos_repost": 0,
            # enqueue 数量（evaluate 的 BFS 只 enqueue type==1）
            "l1_enqueued": 0,
            "l2_enqueued": 0,
        }
        while queue and (node_limit <= 0 or steps_done < node_limit):
            curr_node = queue.popleft()
            curr_depth = int(curr_node.get("_step_depth", 0))

            rm = curr_node.get("reward_model") or {}
            cond_gt = rm.get("ground_truth", {}).get("cond_gt_by_turn") or []
            edge_types_by_turn = rm.get("ground_truth", {}).get("edge_types_by_turn") or []
            curr_gold = []
            gold_info_by_user = {}

            if curr_depth < len(cond_gt):
                layer_gt = cond_gt[curr_depth]
                parent_name = curr_node.get("user_name") or ROOT_FALLBACK_KEY
                found = False
                for item in layer_gt:
                    if parent_name in item:
                        curr_gold = item[parent_name]
                        found = True
                        break
                if (not found) and curr_depth > 0:
                    # 保留原逻辑（不做额外改动）
                    for item in layer_gt:
                        if ROOT_FALLBACK_KEY in item:
                            pass

            # build gold info map for ROUGE/ACC (gold_type/gold_text)
            if curr_depth < len(edge_types_by_turn):
                layer_edges = edge_types_by_turn[curr_depth] or []
                parent_name = curr_node.get("user_name") or ROOT_FALLBACK_KEY
                for item in layer_edges:
                    if parent_name in item:
                        edges = item.get(parent_name) or []
                        for e in edges:
                            if not isinstance(e, dict):
                                continue
                            uname = e.get("user_name")
                            if not uname:
                                continue
                            gold_info_by_user[uname] = {
                                "gold_type": int(e.get("gold_type", 0) or 0),
                                "gold_text": str(e.get("gold_text", "") or ""),
                            }
                        break

            all_preds = []

            if curr_depth == 0:
                # 对生成式评估：优先用 GlobalPoolManager 聚合出来的全量候选池
                all_candidates = pool_mgr.get_candidates(record_id)
                if not all_candidates:
                    all_candidates = smart_extract_potentials(curr_node)
                if not all_candidates:
                    all_candidates = extract_potentialspan_from_text(curr_node.get("content", ""))

                # root 层候选池可能很大：允许按 chunk rollout（只为统计树规模/扩展队列；不做严格顺序对齐）
                use_chunk = int(root_chunk_size or 0)
                if use_chunk <= 0:
                    use_chunk = DYNAMIC_CHUNK_SIZE

                if (not root_chunk_size) or len(all_candidates) <= use_chunk:
                    preds, chat_text, raw_gen = step_rollout(
                        model,
                        tokenizer,
                        device,
                        sys_prompt,
                        curr_node.get("content", ""),
                        all_candidates,
                        curr_depth,
                        gen_params,
                    )
                    all_preds = preds
                    if f_io:
                        f_io.write(
                            json.dumps(
                                {
                                    "ts": int(time.time() * 1000),
                                    "_event": "io",
                                    "_seq": int(seq),
                                    "_ridx": int(ridx),
                                    "group_id": record_id,
                                    "depth": curr_depth,
                                    "input_text": chat_text,
                                    "output_text": raw_gen,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    # chunk rollout; merge by user_name (prefer positive; keep first non-empty content)
                    merged: Dict[str, Dict[str, Any]] = {}
                    chunks = [all_candidates[i:i + use_chunk] for i in range(0, len(all_candidates), use_chunk)]
                    for chunk_idx, chunk_pots in enumerate(chunks):
                        chunk_preds, chat_text, raw_gen = step_rollout(
                            model,
                            tokenizer,
                            device,
                            sys_prompt,
                            curr_node.get("content", ""),
                            chunk_pots,
                            curr_depth,
                            gen_params,
                        )
                        for p in chunk_preds:
                            uname = p.get("user_name")
                            if not uname:
                                continue
                            prev = merged.get(uname)
                            if prev is None:
                                merged[uname] = p
                            else:
                                # upgrade type if needed (positive beats 0; comment/repost both positive)
                                try:
                                    t_new = int(p.get("type", 0))
                                except Exception:
                                    t_new = 0
                                try:
                                    t_old = int(prev.get("type", 0))
                                except Exception:
                                    t_old = 0
                                if (t_old == 0) and (t_new in (1, 2)):
                                    prev["type"] = t_new
                                # fill content if empty
                                if (not prev.get("content")) and p.get("content"):
                                    prev["content"] = p.get("content")
                        if f_io:
                            f_io.write(
                                json.dumps(
                                    {
                                        "ts": int(time.time() * 1000),
                                        "_event": "io",
                                        "_seq": int(seq),
                                        "_ridx": int(ridx),
                                        "group_id": record_id,
                                        "depth": curr_depth,
                                        "chunk_idx": chunk_idx,
                                        "input_text": chat_text,
                                        "output_text": raw_gen,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    all_preds = list(merged.values())

            else:
                all_candidates = pool_mgr.get_candidates(record_id)
                if not all_candidates:
                    all_candidates = extract_potentialspan_from_text(curr_node.get("content", ""))

                chunks = [
                    all_candidates[i:i + DYNAMIC_CHUNK_SIZE]
                    for i in range(0, len(all_candidates), DYNAMIC_CHUNK_SIZE)
                ]
                for chunk_idx, chunk_pots in enumerate(chunks):
                    chunk_preds, chat_text, raw_gen = step_rollout(
                        model,
                        tokenizer,
                        device,
                        sys_prompt,
                        curr_node.get("content", ""),
                        chunk_pots,
                        curr_depth,
                        gen_params,
                    )
                    all_preds.extend(chunk_preds)
                    if f_io:
                        f_io.write(
                            json.dumps(
                                {
                                    "ts": int(time.time() * 1000),
                                    "_event": "io",
                                    "_seq": int(seq),
                                    "_ridx": int(ridx),
                                    "group_id": record_id,
                                    "depth": curr_depth,
                                    "chunk_idx": chunk_idx,
                                    "input_text": chat_text,
                                    "output_text": raw_gen,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            positive_preds = [p for p in all_preds if p["type"] in (1, 2)]
            # 规模统计：L1/L2
            if curr_depth == 0:
                tree_stats["l1_pos_total"] += len(positive_preds)
                tree_stats["l1_pos_comment"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 1)
                tree_stats["l1_pos_repost"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 2)
                tree_stats["l1_enqueued"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 1)
            elif curr_depth == 1:
                tree_stats["l2_pos_total"] += len(positive_preds)
                tree_stats["l2_pos_comment"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 1)
                tree_stats["l2_pos_repost"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 2)
                tree_stats["l2_enqueued"] += sum(1 for p in positive_preds if int(p.get("type", 0)) == 1)

            out_list = []
            for p in all_preds:
                uname = p.get("user_name")
                info = gold_info_by_user.get(uname, {})
                out_list.append(
                    {
                        "user_name": uname,
                        "type": p.get("type", 0),
                        "pred_type": p.get("type", 0),
                        "content": p.get("content", ""),
                        "gold_type": info.get("gold_type", 0),
                        "gold_text": info.get("gold_text", ""),
                    }
                )
            f_det.write(
                json.dumps(
                    {
                        "ts": int(time.time() * 1000),
                        "_event": "node",
                        "_seq": int(seq),
                        "_ridx": int(ridx),
                        "group_id": record_id,
                        "input_user": {"user_name": curr_node.get("user_name")},
                        "depth": curr_depth,
                        "output_text": out_list,
                        "gold": curr_gold,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if curr_depth < depth_limit:
                next_depth = curr_depth + 1
                for p in positive_preds:
                    if p["type"] == 1:
                        child_name = p["user_name"]
                        if not child_name:
                            continue
                        child_interests = pool_mgr.find_interests(record_id, child_name)
                        root_ctx = json.dumps(
                            {"root_username": root_username, "root_content": pure_root_content},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        new_content_base = (
                            f"username: {child_name}\ncontent:\n{p['content']}\n"
                            f"userinterest: {json.dumps(child_interests, ensure_ascii=False)}\n"
                            f"root_context: {root_ctx}"
                        )
                        queue.append(
                            {
                                "user_name": child_name,
                                "content": new_content_base,
                                "depth": next_depth,
                                "_step_depth": next_depth,
                                "reward_model": curr_node.get("reward_model"),
                            }
                        )
            steps_done += 1

        # Mark this record as fully completed for resume logic.
        f_det.write(
            json.dumps(
                {
                    "ts": int(time.time() * 1000),
                    "_event": "record_done",
                    "_seq": int(seq),
                    "_ridx": int(ridx),
                    "group_id": record_id,
                    "tree_stats": tree_stats,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if f_io:
            f_io.write(
                json.dumps(
                    {
                        "ts": int(time.time() * 1000),
                        "_event": "record_done",
                        "_seq": int(seq),
                        "_ridx": int(ridx),
                        "group_id": record_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        processed_since_resume += 1
        if processed_since_resume % 10 == 0:
            f_det.flush()
            if f_io:
                f_io.flush()

    if rank == 0:
        iterator.close()

    f_det.close()
    if f_io:
        f_io.close()
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("[Done] All GPUs finished evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input .jsonl file")
    parser.add_argument("--model", required=True, help="Base model path (SFT ckpt)")

    # 兼容旧参数：checkpoint_pt 可以是 .pt 文件，也可以是 LoRA adapter 目录
    parser.add_argument("--checkpoint_pt", default=None, help="Optional: .pt state_dict OR LoRA adapter dir")
    # 推荐新参数：明确告诉脚本这是 LoRA adapter 目录
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter directory (contains adapter_config.json)")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base weights for faster inference")

    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--jsonl_detail", required=True, help="Output prefix")
    parser.add_argument("--jsonl_io", default=None, help="Output prefix")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=0,
        help="Max nodes (BFS steps) processed per record. 0=unlimited (fully expand).",
    )
    parser.add_argument("--depth_limit", type=int, default=1)
    parser.add_argument(
        "--group_by_record_id",
        action="store_true",
        help="Group evaluation by sft_chunk_info.record_id: multiple lines with same record_id are treated as one record (useful for chunked candidate pools).",
    )
    parser.add_argument(
        "--root_chunk_size",
        type=int,
        default=0,
        help="Optional: chunk rollout for root candidates if pool is large. 0=disable chunking (single rollout).",
    )
    parser.add_argument(
        "--candidate_chunk_size",
        type=int,
        default=50,
        help="Chunk size for candidates when depth>0. Smaller is safer for context but slower; larger is faster but may hit token limits.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing *.rank{rank} logs: truncate incomplete tail across all ranks and append from the next record.",
    )
    args = parser.parse_args()

    evaluate_jsonl(
        args.data,
        args.model,
        args.checkpoint_pt,
        args.tokenizer,
        args.jsonl_detail,
        args.jsonl_io,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        depth_limit=args.depth_limit,
        adapter_dir=args.adapter_dir,
        merge_lora=args.merge_lora,
        group_by_record_id=args.group_by_record_id,
        root_chunk_size=args.root_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
        resume=args.resume,
    )
