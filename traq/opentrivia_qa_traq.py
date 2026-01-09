from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer
from openai import OpenAI
from skopt import gp_minimize
from skopt.space import Real

# PROMPTS from Appendix D.3 of the paper
ZERO_SHOT_PROMPT = """Answer the following question based on the given context; Answer the question shortly.
Question: {question}
Context: {context}
Answer:"""

FEW_SHOT_PROMPT = """Answer the following question based on the given context; Answer the question shortly.
Question: {question1}
Context: {context1}
Answer: {answer1}
Question: {question2}
Context: {context2}
Answer: {answer2}
Question: {question}
Context: {context}
Answer:"""


def make_zero_shot_prompt(question: str, context: str) -> str:
    return ZERO_SHOT_PROMPT.format(question=question, context=context)


def make_few_shot_prompt(
    question: str,
    context: str,
    demo1: Dict[str, str],
    demo2: Dict[str, str],
) -> str:
    return FEW_SHOT_PROMPT.format(
        question1=demo1["q"], context1=demo1["ctx"], answer1=demo1["a"],
        question2=demo2["q"], context2=demo2["ctx"], answer2=demo2["a"],
        question=question, context=context,
    )


# Data Model
@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]

@dataclass
class Passage:
    pid: str
    title: str
    text: str
    score: float

@dataclass
class SplitSpec:
    calib_qids: List[str]
    test_qids: List[str]
    opt_qids: List[str]
    cal_qids: List[str]
    p_star: Dict[str, str]  # qid -> pid



# Load OpenTriviaQA from HuggingFace
def load_opentrivia_qa(seed = 0):
    ds = load_dataset("rlyapin/OpenTriviaQA", "general")
    rows = ds["train"]
    out: List[QAExample] = []
    for row in rows:
        out.append(QAExample(
        qid=str(row["id"]),
            question=str(row["question"]),
            answers=[str(row["answer"])],
        ))
    random.Random(seed).shuffle(out)
    return out
