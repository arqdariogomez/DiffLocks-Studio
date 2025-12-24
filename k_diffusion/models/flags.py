import os
import torch

# --- 1. COMPILATION (JIT) ---
# Disabled to avoid "INTERNAL ASSERT FAILED" crash on T4
use_compile = False

def get_use_compile():
    return use_compile

def compile_wrap(function):
    return function

# --- 2. CHECKPOINTING (Gradient Checkpointing) ---
_checkpointing = False

def checkpointing(enable=True):
    global _checkpointing
    _checkpointing = enable

def get_checkpointing():
    global _checkpointing
    return _checkpointing

# --- 3. FLASH ATTENTION 2 ---
# Controls if FA2 is attempted. Set to False by default for safety,
# as we have our own manual fallback in attention.py.
_use_flash_attention_2 = False

def use_flash_attention_2(enable=True):
    global _use_flash_attention_2
    _use_flash_attention_2 = enable

def get_use_flash_attention_2():
    global _use_flash_attention_2
    return _use_flash_attention_2
