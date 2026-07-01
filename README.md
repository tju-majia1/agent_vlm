# Mobile GUI Agent

一个**操作真实 Android 手机的 vision-grounded GUI agent**：用一句自然语言描述任务，
agent 自己看屏幕截图、用视觉大模型（VLM）推理"该点哪 / 输什么"、用 adb 控制手机。

```bash
python run_agent.py "打开微信，给张三发『我到了』"
python run_agent.py "打开设置，开启深色模式"
python run_agent.py "打开备忘录，新建一条：明天交作业"
```

---

## 1. 架构

```
        用户任务（自然语言）
                │
                ▼
  ┌──────────────────────────────────────────┐
  │   GUIAgent.run() 主循环                     │
  │                                            │
  │   while action_budget < max_steps:         │
  │       img = adb screencap                   │← 真机截图
  │       [可选] 抓 UI 树给可点元素叠红框+编号    │← Set-of-Marks
  │       decision = vlm(prompt, img)           │← VLM 推理（出 JSON 决策）
  │       executor.do(decision)                 │← adb input tap/text/swipe/...
  │       if decision.done: break               │
  │       # wait 不消耗 action_budget           │
  └──────────┬───────────────┬───────────────┘
             │               │
             ▼               ▼
        AdbExecutor    VLM (Qwen3-VL / Claude / GPT-4o)
        - screenshot   - 看图 + 任务 → JSON 决策
        - tap / tap_id   {"action":"tap","x":540,"y":1200,"done":false}
        - type / clear   {"action":"tap_id","id":7,"done":false}
        - enter / back / swipe / long_press / open_app / wait
```

**两种 grounding 模式（用 `USE_SOM` 切换）**
- **Set-of-Marks（默认，`USE_SOM=1`）**：先抓无障碍树，把**可点元素**叠上红框+编号，
  让模型按编号 `tap_id` 选，按元素中心点 —— 标准 app（设置/Play 等）几乎不偏。
  树为空（画布/WebView/游戏）自动退回纯视觉。
- **纯视觉（`USE_SOM=0`）**：完全不碰 UI 树，模型自估 `[0,1000]` 坐标。任何界面都能点，
  精度看模型（`qwen3-vl-plus` 这类 GUI 专用模型在纯视觉下明显更准）。

**核心代码**：
- `mobilerun/agent.py`   vision-grounded 主循环 + Set-of-Marks 标注 + JSON 解析/容错 +
                          归一化坐标转换 + 死循环/卡死检测 + wait 预算 + 截图落盘
- `mobilerun/executor.py` 自研 ADB 真机执行器（adb 自动定位 + 瞬时错误重试）
- `mobilerun/llm.py`     OpenAI 兼容多模态客户端（含网络重试）

**可选加速模块** `mobilerun/skills/`（参数化技能库）：
- 跑成功的 trace 蒸馏成参数化技能（contact / message 等 slot）
- 语义召回：embedding + cosine（真·余弦，不依赖后端是否归一化）
- 自愈：UI 改版时文本相似度 / LLM 同义词重定位
- 接不接都不影响 agent 跑通，是单纯的"重复任务加速器"

> **闭环现状（重要）**：默认主路径 `run_agent.py` 只跑 vision agent，不碰技能库。
> 设 `LEARN_SKILLS=1` 后，成功的任务会自动蒸馏成技能并写入 `data/skills.json`
> （extract → index → store 这半条闭环已接通；坐标用执行时换算出的真实设备像素，
> 见 `GUIAgentResult.trace_for_skill()`）。**召回后自动回放**仍是手动环节：
> 用 `SkillRunner`（`mobilerun/skills/runner.py`）自行串，尚未在主入口默认启用。

---

## 2. 跑起来

### 2.1 装

```powershell
cd D:\gui_agent\mobilerun
pip install -e .
adb version       # 可选：adb 不在 PATH 也行（见下）
```

依赖只有 `pydantic + requests + pillow`。

> **adb 自动定位**：executor 会按 `ADB_PATH` 环境变量 → PATH → 常见 SDK 路径
> （`%LOCALAPPDATA%\Android\Sdk\platform-tools`、`ANDROID_HOME` 等）依次查找，
> 所以**不把 platform-tools 加进 PATH 也能跑**。找不到时再设 `ADB_PATH` 指向 `adb.exe`。

### 2.2 配 VLM

在项目根目录创建 `.env`。两类后端任选（都走 OpenAI 兼容协议）：

```ini
# 方案 A：通义千问 DashScope（推荐纯视觉，原生 [0,1000] 坐标，grounding 准）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
QWEN_CHAT_MODEL=qwen3-vl-plus

# 方案 B：OpenAI / 任意 OpenAI 兼容网关（如 Claude 网关）
OPENAI_API_KEY=sk-xxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1     # 网关要带 /v1
OPENAI_CHAT_MODEL=gpt-4o                       # 或 claude-sonnet-4-6 等
```

两个 key 都填时，用 `LLM_PREFER` 选（`qwen` / `openai` / `auto`）。
`.env` 加载逻辑写在 `run_agent.py` 顶部，无需 `python-dotenv`。

### 2.2.1 运行开关（环境变量）

| 变量 | 默认 | 作用 |
|---|---|---|
| `LLM_PREFER` | `auto` | 选后端：`qwen`（DashScope）/ `openai`（OpenAI 兼容）/ `auto`（OPENAI 优先） |
| `USE_SOM` | `1` | `1`=Set-of-Marks（UI 树标注）；`0`=纯视觉坐标（不碰 UI 树） |
| `MAX_STEPS` | `20` | **动作步数**上限（wait 不计入） |
| `BLOCK_RISKY` | `0` | `1`=拦截卸载/支付/删除/发送等不可逆动作（命中即停，不执行） |
| `SCREENS_DIR` | `data/screens` | 截图落盘目录（每次 run 建带时间戳子目录） |
| `LEARN_SKILLS` | `0` | `1`=成功后把轨迹蒸馏成技能写入 `data/skills.json`（见 §1 闭环说明） |
| `ADB_PATH` | 自动 | 显式指定 `adb.exe`；不设则自动找 PATH / 常见 SDK 路径 |

例：
```bash
# 纯视觉 + 通义千问
LLM_PREFER=qwen USE_SOM=0 python run_agent.py "打开设置，开启深色模式"
# Set-of-Marks + Claude 网关 + 风险护栏
LLM_PREFER=openai USE_SOM=1 BLOCK_RISKY=1 python run_agent.py "卸载微信"
```

### 2.3 手机准备

1. 开发者选项 → 打开 USB 调试
2. 数据线插 PC，弹"允许 USB 调试吗"勾允许
3. `adb devices` 能看到设备且状态 `device`
4. （可选，中文输入需要）装 [ADBKeyBoard.apk](https://github.com/senzhk/ADBKeyBoard)：
   ```powershell
   adb install ADBKeyboard.apk
   adb shell ime enable com.android.adbkeyboard/.AdbIME
   adb shell ime set com.android.adbkeyboard/.AdbIME
   ```
   > **模拟器额外一步**：AVD 自带硬件键盘时，软 IME（AdbIME）不会启动，其广播接收器
   > 不注册 → 中文注入静默失败（广播 result=0 但没字进框）。需开启"硬键盘下也显示软键盘"：
   > ```powershell
   > adb shell settings put secure show_ime_with_hard_keyboard 1
   > ```
   > AdbIME 没有可见按键是正常的；agent 直接走 broadcast 注入，不依赖可见键盘。

### 2.4 跑

```powershell
python run_agent.py "打开设置，开启深色模式"
```

日志大致是：

```
设备: emulator-5554  分辨率: 1080x2424
模型: OpenAICompatibleLLM  (qwen3-vl-plus)
模式: Set-of-Marks (UI树标注)
=== Task: 打开设置，开启深色模式 ===
--- Step 1 ---
[Observe] image 1080x2424  scale=1.000
[SoM]     标注 24 个可交互元素
[Observe] 已保存截图 data/screens/run_20260630_220307/step_01.png
[Plan]    打开系统设置应用
[Plan]    {'action': 'open_app', 'package': 'com.android.settings'}
[Act]     OK open com.android.settings
--- Step 2 ---
[Observe] image 1080x2424  scale=1.000
[Plan]    点击 Display & touch
[Plan]    {'action': 'tap_id', 'id': 8}
[Act]     OK tap_id 8 -> (410,1934)
...
=== Done after 6 steps ===
========== 结果 ==========
  success     : True
  steps       : 6
  stop_reason : done
```

每次 run 的截图保存在 `data/screens/run_<时间戳>/step_NN.png`（带 SoM 标注，
即"模型实际看到的那张"），方便逐帧复盘、对照模型说的坐标/编号和实际页面。

---

## 3. 目录结构

```
mobile-gui-agent/
├── run_agent.py                   ← 主入口
├── README.md
├── pyproject.toml                 deps: pydantic + requests + pillow
├── .env                           本地 API key（.gitignore 已忽略）
├── mobilerun/
│   ├── __init__.py                exports GUIAgent / AdbExecutor / build_llm_from_env
│   ├── agent.py                   GUIAgent vision-grounded 主循环
│   ├── executor.py                AdbExecutor (screencap / tap / type / swipe)
│   ├── llm.py                     OpenAI 兼容多模态客户端
│   ├── schema.py                  SkillStep / Slot 等数据结构
│   ├── self_heal.py               (skill 自愈用，agent 主路径不依赖)
│   └── skills/                    （可选）参数化技能库
│       ├── store.py
│       ├── retriever.py
│       ├── extractor.py
│       ├── filler.py
│       └── runner.py
├── tests/test_agent_core.py             agent 单测（JSON容错/坐标/帧哈希/动作归一/SoM/wait预算/护栏）
├── tests/test_executor_core.py          executor 单测（输入转义/bounds/UI树解析/adb定位）
├── tests/test_retriever_cosine.py       余弦相似度单测
├── tests/skills/test_skill_library.py   技能库单测（全套共 62 个测试，pytest 跑）
└── data/                          运行时产生（截图、技能、状态图）
```

---

## 4. 关键实现要点

### 4.1 归一化坐标 [0, 1000]
Qwen3-VL / Qwen2-VL 等现代 VLM 用 `[0, 1000]` 归一化坐标系（图像左上 = (0,0)，
右下 = (1000,1000)）。`GUIAgent._to_device_px()` 自动把模型输出乘以设备分辨率比例
得到真实像素坐标。对老模型（如 `qwen-vl-max` 输出**缩放后截图**的绝对像素）会自动
退到兼容模式，并按 `scale`(=设备像素/截图像素) 还原回设备坐标。

### 4.2 JSON 容错
VLM 偶尔会出小毛病的 JSON（数字后多余引号、漏键名、末尾逗号、外面包代码块）。
`GUIAgent._repair_json()` 自动修复这几类常见错误后再交给严格解析；首次失败还会附一
条"请只返回 JSON"的重试。

### 4.3 死循环 / 卡死检测
两条独立信号：
1. **同坐标重复**：连续 3 次完全相同的 `tap`（含坐标）就 halt，导航类动作
   （back / swipe / wait）允许合法连击不触发。
2. **屏幕停滞**：把每帧截图压成 16×16 灰度量化哈希（`_frame_hash`，容忍时钟/
   光标闪烁），连续 3 次"动作后屏幕没变"也 halt —— 能抓到同坐标检测漏掉的
   A→B→A 震荡和"点了但没反应"。`wait` 不计入停滞。

### 4.4 中文输入
`adb shell input text` 原生只吃 ASCII。装了 ADBKeyBoard 后，`executor.type_text()`
自动识别中文 / ASCII 走对应路径，中文通过 broadcast 走 **base64 通道（ADB_INPUT_B64）**
注入，绕开所有 shell 引号/特殊字符问题；ASCII 路径也对 `'`、`&`、`(` 等元字符做转义。

### 4.5 Set-of-Marks（`USE_SOM=1`）
抓无障碍树里的**可点元素**，在截图上叠红框+编号，prompt 附编号清单，模型用
`{"action":"tap_id","id":N}` 选 → 点该元素中心。消掉密集界面上"点偏到相邻控件/空白"
的 grounding 误差（如设置开关、底部 tab、列表项）。树为空（画布/WebView/游戏）自动
退回纯视觉坐标。UI 树用 `uiautomator dump` + `exec-out cat` 直读内存（无临时文件）。

### 4.6 动作集与同义词归一
动作：`tap` / `tap_id` / `long_press` / `type`（可带 `"clear": true` 先清空）/
`clear` / `enter`（回车提交搜索）/ `swipe` / `back` / `open_app` / `wait` / `finish`。
不同 VLM 用词不同（Qwen 爱写 `click`/`scroll`/`input`），`_normalize_action` 统一
映射到上面词汇（`click→tap`、`scroll→swipe`、`input→type`、`go_back→back`…）。

### 4.7 wait 预算 & 长下载
`wait` **不消耗 `max_steps` 动作预算**，单独设上限（连续 25 / 累计 40 次），且连续
wait 睡眠递增（1.5s→封顶 8s）。等大文件下载/长加载时既不挤占动作步数，也大幅减少
VLM 调用次数；真卡住则触发"等待超时"停止。

### 4.8 健壮性
- **adb 自动定位**：`ADB_PATH` → PATH → 常见 SDK 路径，免配 PATH。
- **adb 瞬时重试**：`_adb_raw` 对超时/设备瞬时离线退避重试；`screenshot()` 重试 + 校验 PNG 头。
- **LLM 网络重试**：`OpenAICompatibleLLM.chat` 对 5xx/429/连接错误指数退避，避免一次抖动毁掉整个任务。
- **`open_app`**：先 `cmd package resolve-activity` 解析 LAUNCHER 组件再 `am start -n`，解析不到回退 monkey。
- **风险护栏（`BLOCK_RISKY=1`）**：tap_id 命中卸载/支付/删除/发送等关键词时告警，开启后直接拦截。

---

## 5. 两种模式实测对比（真机 emulator-5554, 1080×2424）

### 5.1 同一难任务：时钟设闹钟（需点模拟时钟**表盘数字** + **AM/PM** 小开关）
这是检验 grounding 精度的硬骨头（细粒度、相邻小目标）。

| 配置 | 表盘"8" | AM/PM | 结果 |
|---|---|---|---|
| Claude Sonnet 4.6 · 纯视觉 | ❌ 点偏 | — | 失败（stall） |
| Claude Opus 4.8 · 纯视觉 | ✅ | ❌ 点到 PM | 失败（loop） |
| Claude · **Set-of-Marks** | ✅ `tap_id` | ✅ `tap_id` | ✅ 6 步 |
| **qwen3-vl-plus · 纯视觉** | ✅ 自估坐标 | ✅ 自估坐标 | ✅ 7 步 |

**结论**：通用推理模型（Claude）纯视觉点不准这种小目标，得靠 Set-of-Marks 拿元素中心；
GUI 专用模型（Qwen3-VL）纯视觉就能打中。换更强的推理模型只能缓解、修不掉根因。

### 5.2 各类任务实测

| 任务 | 模式 / 模型 | 结果 | 备注 |
|---|---|---|---|
| 打开时钟 app | SoM / Claude | ✅ 2 步 | |
| 设置里搜"蓝牙"（中文输入） | SoM / Claude | ✅ 5 步 | ADBKeyBoard base64 注入，逐字正确 |
| 时钟设 8:00 闹钟 | SoM / Claude | ✅ 6 步 | 表盘+AM/PM 全靠 `tap_id` |
| Play 搜索并安装 QQ | SoM / Claude | ✅ 15 步 | **全自主**：搜索→type→点结果→Install |
| 下载抖音（=TikTok 国际版） | 纯视觉 / Qwen3-VL | ✅* | 搜索/中文输入/点 Install 全对；182MB 大文件后台装完（*触发安装成功，等待超步数） |
| 打开"深色模式" | 纯视觉 / Qwen3-VL | ✅ 11 步 | 开抽屉→Settings→Display→开关，中途纠错 |
| 打开"明亮模式" | 纯视觉 / Qwen3-VL | ⚠️ | 坐标点对了，但模型**读不准开关开/关状态**反复切，被 loop 拦下（末态恰为 light）|

> 失败/降级案例都不是 harness 的锅：循环/卡死/wait 超时检测都正确止损。
> 纯视觉的两类边界是 **细粒度 grounding**（小目标）和 **状态感知**（开关是开是关），
> 前者用 SoM 或 Qwen3-VL 解决，后者目前仍是纯视觉的弱项。

---

## 6. FAQ

**Q: 坐标怎么换算的？**
A: 模型输出 `[0, 1000]` 归一化坐标 → `device_x = x * device_width / 1000`。
   截图缩放与设备像素是两套独立坐标系，由 `_resize_png()` 和 `_to_device_px()`
   分别管，互不干扰。

**Q: 纯视觉 和 Set-of-Marks（UI 树+编号）该用哪个？**
A: 两种都内置，`USE_SOM` 切换。
   - **Set-of-Marks**：标准 app（设置/Play/系统应用）点击最准（点元素中心，几乎不偏），
     但依赖无障碍树，画布/WebView/游戏覆盖不到（会自动退回纯视觉）。
   - **纯视觉**：任何界面都能点，不碰 UI 树；精度看模型，`qwen3-vl-plus` 这类 GUI 专用
     模型在 ~1080p 截图上的纯视觉 grounding 已能打中表盘数字、AM/PM 这类小目标。
   经验：标准 app 用 SoM 更稳；画布/游戏/WebView 或想完全脱离无障碍树时用纯视觉 + Qwen3-VL。

**Q: 中文输入怎么办？**
A: 见 4.4。装 ADBKeyBoard，executor 自动判断走 broadcast 注入。

**Q: 跑一次大概多少 token？**
A: 截图缩到 1280 宽再发，`qwen3-vl-plus` 每步约 1.5-2k input + 100 output。
   10 步任务 ≈ 20k tokens。

**Q: 不止微信，其他 app 也能跑？**
A: 能。Agent 直接看截图 + 操控 adb，对任何 app 都通用。`open_app` 需要知道包名
   （如 `com.android.settings`），也可以让模型先 tap 桌面图标进入。
