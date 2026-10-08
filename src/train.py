"""比較相手: 1 件を SciGym 公式 Controller（ReAct）で解かせる。公式の履歴はクラス属性なので 1 件 1 プロセス。"""

import ctypes
import json
import os
import sys
import sysconfig
import time
from pathlib import Path

# libroadrunner は libpython を動的に探すので、ローダのパスに無い環境（uv 管理の Python）では先読みする
_libpython = f"{sysconfig.get_config_var('LIBDIR')}/libpython3.11.so.1.0"
if os.path.exists(_libpython):
    ctypes.CDLL(_libpython, mode=ctypes.RTLD_GLOBAL)

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402
from scigym.api import LLM  # noqa: E402
from scigym.controller import Controller  # noqa: E402

BUDGET_EXCEEDED = 42  # 予算超過（HTTP 402）。Gateway の予算が尽きたら main.py は run 全体を止める（件が欠けた run を残さない）
CONTEXT_OVERFLOW = 43  # 会話が文脈長を超えた。main.py はやり直さず「提出に至らなかった件」として数える


class OpenAICompatible(LLM):
    """公式の scigym.agent.GPT と同じ手順で、OpenAI 互換エンドポイントを呼ぶ。"""

    def initialize(self, base_url):
        self.client = OpenAI(
            api_key=os.environ["VERCEL_AI_GATEWAY_API_KEY"], base_url=base_url, max_retries=1, timeout=1800
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages

    def get_response(self, user_message):
        self.add_message("user", user_message)
        max_tokens = self.max_length
        for attempt in range(30):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                if attempt == 29:
                    raise
                print(f"connection error ({type(exc).__name__}); retry {attempt + 1} after 60s", flush=True)
                time.sleep(60)
                continue
            except openai.APIStatusError as exc:
                if exc.status_code == 402:
                    sys.exit(BUDGET_EXCEEDED)
                if exc.status_code == 400 and any(w in str(exc).lower() for w in ("context", "maximum", "too long", "max_tokens")):
                    print(f"context overflow after {len(self.messages)} messages: {exc}", flush=True)
                    sys.exit(CONTEXT_OVERFLOW)
                if exc.status_code < 500 or attempt == 29:
                    raise
                print(f"gateway {exc.status_code}; retry {attempt + 1} after 60s", flush=True)  # 上流の一時的な不調は待って呼び直す
                time.sleep(60)
                continue
            choice = response.choices[0]
            text = choice.message.content
            if isinstance(text, str) and len(text) > 0:
                break
            usage = response.usage.model_dump() if response.usage else None
            print(f"empty response: finish_reason={choice.finish_reason} max_tokens={max_tokens} usage={usage}", flush=True)
            if choice.finish_reason == "length":  # thinking で使い切った。上限を上げて呼び直す
                max_tokens = min(max_tokens * 2, 131072)
            else:  # 本文が空で reasoning 側で終わっている応答は、コードブロックまで reasoning に入っている。その文を本文として返す
                reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
                if isinstance(reasoning, str) and len(reasoning) > 0:
                    text = reasoning
                    break
        assert isinstance(text, str) and len(text) > 0, "empty response"
        self.add_message("assistant", text)
        usage = response.usage
        self.input_total_tokens += usage.prompt_tokens if usage else 0
        self.output_total_tokens += usage.completion_tokens if usage else 0
        return text, {}


def main():
    cfg = json.loads(sys.argv[1])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    controller = Controller(
        path_to_sbml_cfg=cfg["instance_dir"],
        max_iterations=cfg["max_iterations"],
        test_memorize=False,
        output_directory=str(out),
        experiment_actions_path="prompts/experiment_actions_perturb.md",
        customized_functions_path="prompts/customized_functions_sim.md",
        eval_debug_rounds=cfg["eval_debug_rounds"],
        temperature=cfg["temperature"],
    )
    llm = OpenAICompatible(
        model_name=cfg["model"],
        api_key="",
        system_prompt=controller._create_system_prompt(),
        temperature=cfg["temperature"],
        max_length=cfg["max_tokens"],
        base_url=cfg["base_url"],
    )
    controller.run_benchmark(model=llm)
    history = json.loads((out / "chat_history.yaml").read_text()) if (out / "chat_history.yaml").exists() else {}
    (out / "tokens.json").write_text(json.dumps({
        "input_tokens": llm.input_total_tokens,
        "output_tokens": llm.output_total_tokens,
        # 正解モデルへの問い合わせ回数（成功した実験）。autoscilab 側の予算と同じ単位
        "n_experiments": sum(1 for v in controller.experiment_request_history.values() if v.get("success")),
        "n_iterations": max(len(history) - 1, 0),  # Iteration 0 は初期プロンプト
        "submitted": (out / "final_model.xml").exists(),
    }))


if __name__ == "__main__":
    main()
