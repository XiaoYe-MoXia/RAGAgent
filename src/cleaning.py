"""5 层数据清洗（Phase 1）

数据流：raw_input → normalize → denoise → filter_sensitive → quality_check → business_rules
"""
import re
import unicodedata
from typing import Tuple

from .config import MAX_LEN, MIN_LEN, SENSITIVE_WORDS


class InputRejected(Exception):
    """清洗阶段判定输入为垃圾/无效时抛出，pipeline 兜底处理"""
    pass


# --- 第1层：标准化 ---
def normalize_text(text: str) -> str:
    """去首尾空格、全角转半角"""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text


# --- 第2层：去噪 ---
def remove_noise(text: str) -> str:
    """去除网址、手机号、微信号、邮箱、重复字等噪声"""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'1[3-9]\d[\d\s-]{8,}\d', '', text)
    text = re.sub(r'(?:wx|vx|微信)\s*[:：]\s*[\w-]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 第3层：敏感词过滤 ---
def filter_sensitive(text: str) -> str:
    """把敏感词替换为 [已过滤]"""
    for word in SENSITIVE_WORDS:
        if word in text:
            text = text.replace(word, "[已过滤]")
    return text


# --- 第4层：质量检测 ---
def quality_check(text: str) -> Tuple[bool, str]:
    """返回 (是否通过, 不通过原因)"""
    if len(text) < MIN_LEN:
        return False, f"内容过短（{len(text)}字）"
    if len(text) > MAX_LEN:
        return False, f"内容过长（{len(text)}字）"
    if not re.search(r'[一-龥a-zA-Z]', text):
        return False, "内容无有效文字"
    if len(text) > 10 and len(set(text.replace(" ", ""))) <= 2:
        return False, "内容重复度过高"
    return True, ""


# --- 第5层：业务规则 ---
def business_rules(text: str) -> Tuple[bool, str]:
    """业务级判断钩子"""
    _ = text
    return True, ""


# --- 主清洗函数 ---
def clean_input(inputs: dict) -> dict:
    """5 层串联清洗"""
    text = inputs.get("raw_input", "")

    text = normalize_text(text)
    if not text:
        raise InputRejected("输入为空")
    text = remove_noise(text)
    text = filter_sensitive(text)

    ok, reason = quality_check(text)
    if not ok:
        raise InputRejected(reason)

    ok, reason = business_rules(text)
    if not ok:
        raise InputRejected(reason)

    return {"input": text}