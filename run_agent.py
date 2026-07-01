"""
项目主入口：vision-grounded GUI Agent

用法：
    python run_agent.py "打开微信，给张三发『我到了』"
    python run_agent.py "打开设置，开启深色模式"
    python run_agent.py "在备忘录里新建一条：明天交作业"

前置条件：
    1. 手机插好 USB 调试，adb devices 能看到
    2. 设环境变量 DASHSCOPE_API_KEY（推荐，通义 VL-Max 便宜）
       或 OPENAI_API_KEY（要用 gpt-4o 这种带视觉的）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(Path(__file__).resolve().parent / ".env")

from mobilerun.agent import GUIAgent
from mobilerun.executor import AdbExecutor
from mobilerun.llm import build_llm_from_env


def _maybe_learn_skill(task: str, result) -> None:
    """可选闭环：成功跑完后把轨迹蒸馏成参数化技能并入库。

    默认关闭，置 LEARN_SKILLS=1 开启。整段包在 try 里，
    任何失败都不影响主任务结果。
    """
    if os.environ.get("LEARN_SKILLS") != "1" or not result.success:
        return
    trace = result.trace_for_skill()
    if not trace:
        return
    try:
        from mobilerun.skills import (
            SkillExtractor, SkillRetriever, SkillStore, build_embedder_from_env,
        )

        llm = build_llm_from_env()  # 蒸馏用文本模型即可
        store = SkillStore(os.environ.get("SKILL_STORE", "data/skills.json"))
        skill = SkillExtractor(llm).extract(task_description=task,
                                            recorded_steps=trace)
        try:
            SkillRetriever(store, build_embedder_from_env()).index_skill(skill)
        except Exception:
            pass  # 没配 embedding 也能存，召回退化到关键词
        store.put(skill)
        print(f"[Learn] 已蒸馏并保存技能 {skill.id!r} → {store.path}")
    except Exception as e:
        print(f"[Learn] 跳过技能学习: {e}")


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("用法: python run_agent.py \"你的任务\"")
        sys.exit(2)
    task = " ".join(sys.argv[1:]).strip()

    executor = AdbExecutor()
    print(f"设备: {executor.serial}  分辨率: {executor.width}x{executor.height}")

    # LLM_PREFER=qwen 走通义千问 DashScope；openai 走 OpenAI 兼容端点；auto 自动
    prefer = os.environ.get("LLM_PREFER", "auto")
    llm = build_llm_from_env(prefer=prefer, vision=True)
    print(f"模型: {llm.__class__.__name__}  ({getattr(llm, 'model', '?')})")

    max_steps = int(os.environ.get("MAX_STEPS", "20"))
    screens_dir = os.environ.get("SCREENS_DIR", "data/screens")
    # USE_SOM=0 关闭 Set-of-Marks（UI 树标注）→ 纯视觉坐标模式
    use_som = os.environ.get("USE_SOM", "1").strip().lower() not in (
        "0", "false", "no", "off")
    # BLOCK_RISKY=1 开启风险动作护栏（拦截卸载/支付/删除/发送等不可逆操作）
    block_risky = os.environ.get("BLOCK_RISKY", "0").strip().lower() in (
        "1", "true", "yes", "on")
    print(f"模式: {'Set-of-Marks (UI树标注)' if use_som else '纯视觉坐标 (无UI树)'}"
          f"{' | 风险护栏:on' if block_risky else ''}")
    agent = GUIAgent(
        executor, llm,
        max_steps=max_steps,
        screens_dir=screens_dir,
        use_som=use_som,
        block_risky_actions=block_risky,
    )
    result = agent.run(task)
    _maybe_learn_skill(task, result)

    print("\n========== 结果 ==========")
    print(f"  success     : {result.success}")
    print(f"  steps       : {len(result.steps)}")
    print(f"  stop_reason : {result.stop_reason}")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
