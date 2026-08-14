# Pi（Python 移植版）

[`@earendil-works/pi`](https://github.com/earendil-works/pi) 的 **Python-native** 移植。
行为对齐原 TypeScript 版（5 个包），但用 Python 习惯写法实现：uv workspace 单体仓库、统一 `EventStream` 协作原语、可插拔 LLM provider、Textual 终端 UI、可插拔 + OTLP 遥测。

> ⚠️ **运行前必读**：内置的 7 个编码工具（含 `bash`）会**直接在宿主机/容器内以当前用户权限执行**，路径限制在 `cwd` 内只是防御性检查，**不是安全边界**。真实（尤其不可信）场景下必须用 Docker 隔离运行——见下文 [Docker 隔离](#docker-隔离运行前提)。

---

## 1. 包结构

| 包 | 职责 |
| --- | --- |
| `pi_ai` | LLM 类型、`EventStream` 原语、6 家 provider 适配（Anthropic / Google / Bedrock / OpenRouter / Qwen / DeepSeek） |
| `pi_agent_core` | `agent_loop` 双循环编排、工具调度/回灌、`length` 自愈、`AgentTool`/`AgentEvent` 契约、默认 `stream_fn` 注册 |
| `pi_coding_agent` | 会话编排 `AgentSession` + 生命周期 `AgentSessionRuntime` + 7 个内置工具 + CLI |
| `pi_tui` | Textual 终端 UI：`interactive` / `print` / `rpc` 三种前端，共用同一 `AgentSession` |
| `pi_telemetry` | 可插拔遥测契约：`noop` / `memory` / `schema` / OTLP 适配 + 一致性校验 harness |

Python ≥ 3.12，依赖统一用 `uv` 管理。

## 2. 安装

```bash
# 克隆后，在仓库根目录
uv sync --all-extras
```

`--all-extras` 会一并安装各包的 optional 依赖：`pi_tui` 的 `textual`、`pi_telemetry` 的 `opentelemetry-*`（api/sdk/exporter-otlp）。
这些 extra 不装的话，`uv run pytest` 会因 import 不到 `textual`/`opentelemetry` 而 collection 失败——**不要只用 `uv sync`**。

## 3. 运行：编码 Agent（CLI）

```bash
# Mock 模式：无需网络/凭证，用罐头回复驱动整轮（agent_loop / 工具调度 / 会话生命周期全跑通）
PI_MOCK=1 uv run python -m pi_coding_agent

# 真实模式：用环境变量指定 provider / 模型，再提供对应凭证（见第 4 节）
PI_API=openai-completions PI_PROVIDER=openrouter OPENROUTER_API_KEY=sk-... uv run python -m pi_coding_agent
```

### 环境变量（CLI 模型配置，`pi_coding_agent/cli.py:build_model`）

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `PI_MOCK` | 置 1 → 走 `mock_stream_fn`（无网络） | 未设 |
| `PI_MODEL_ID` | 模型 id（如 `claude-3-5-sonnet`、`deepseek-chat`） | `claude-3-5-sonnet` |
| `PI_MODEL_NAME` | 展示用名称 | `default` |
| `PI_API` | 调度键，见下表 | `anthropic-messages` |
| `PI_PROVIDER` | provider 名 | `anthropic` |
| `PI_BASE_URL` | 自定义 base URL（自托管/网关） | 空 |

`PI_API` 取值 → provider 适配器（`pi_ai/stream.py`）：

- `anthropic-messages`
- `google-generative-ai`
- `bedrock-converse-stream`
- `openai-completions`（OpenRouter / DeepSeek / Qwen 共用此引擎）

## 4. 凭证（LLM 密钥）约定

凭证解析位置在 LLM 边界（`options.apiKey` 或各 provider 的 env）：

| Provider | 凭证来源 |
| --- | --- |
| OpenRouter | 环境变量 `OPENROUTER_API_KEY`（`_openai_compat.PROVIDER_DEFAULTS`） |
| DeepSeek | 环境变量 `DEEPSEEK_API_KEY` |
| Qwen（DashScope） | 环境变量 `DASHSCOPE_API_KEY` |
| Bedrock | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` / `AWS_REGION`（经 `_env()` 读取） |
| Anthropic / Google | **必须通过 `options.apiKey` 在调用处注入**（`pi_ai` 当前没有从环境变量读 Anthropic/Google key 的逻辑） |

> **已知限制**：CLI 的 `build_model()` 只读取 provider/模型，**不从环境变量读 Anthropic/Google 的 key**。因此——
> - 想用 OpenAI 兼容网关（OpenRouter/DeepSeek/Qwen）：直接设对应 `PI_*` + key 即可，已通。
> - 想用原生 Anthropic/Google：需以编程方式构造 `StreamOptions(apiKey=...)` 注入，或经 OpenAI 兼容网关转发。
> 这一限制不影响 mock 模式与 OpenAI 兼容 provider，是后续可补的点。

## 5. 运行：终端 UI（`pi_tui`）

三种前端驱动同一个 `AgentSession`，行为一致：

```bash
# 交互式（Textual）：Header + 聊天日志 + 输入框，token 级流式渲染
uv run python -m pi_tui --mode interactive

# 打印模式：事件流写到 stdout（可管道喂多行 --message，或走 stdin）
uv run python -m pi_tui --mode print --mock --message "列出当前目录的文件"
echo "hello" | uv run python -m pi_tui --mode print --mock

# RPC 模式：NDJSON 命令/事件协议（host 进程可读可远程驱动）
printf '{"type":"user","text":"hi"}\n{"type":"quit"}\n' | uv run python -m pi_tui --mode rpc --mock
```

通用参数：`--cwd <dir>`（工具工作目录，默认 `.`）、`--mock`（强制 mock LLM）。

## 6. 遥测（`pi_telemetry`）

默认 **noop**，零开销；仅在显式接入时才有开销。

- `NOOP_TELEMETRY_CONTEXT`：默认空实现。
- `InMemoryTelemetryContext`：记录 span / 父子 / 结束顺序，用于测试与本地观测。
- `OtlpTelemetryContext`：懒加载 OpenTelemetry，**未配置时自动回退 noop**（不会因缺依赖崩溃）。
- `define_telemetry_schema` / `create_typed_span_starter`：Pythonic 的 schema 元数据声明（仅元数据，不改运行时语义）。
- `testing/conformance.create_telemetry_adapter_conformance`：忠实改编上游 9 条一致性不变量，新适配实现喂 `TelemetryAdapterFixture` 即可复用。

## 7. Docker 隔离运行前提（重要）

### 为什么必须隔离

`pi_coding_agent` 的 7 个内置工具（`read` `write` `edit` `bash` `grep` `find` `ls`）由 `tools.py` 直接 subprocess/文件系统调用实现：

- `_safe_path(cwd, p)` 只做"拒绝逃出 cwd"的**防御性**检查，**可被 `bash` 工具绕过**（bash 能执行任意命令、读任意环境变量、访问网络）。
- 换言之，没有外部沙箱时，agent 拥有**当前用户的全部权限**——误删文件、读密钥、外联都无从防备。

因此：**真实使用（尤其是让 agent 碰不可信输入/联网）必须跑在 Docker 里**，把可写面收敛到单个工作目录、降权、禁提权。

### 镜像（`Dockerfile`，仓库根）

```dockerfile
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

# 运行期依赖 + optional extras（textual / opentelemetry-otlp），
# 但跳过 dev 组（pytest / ruff）以瘦身。
RUN uv sync --all-extras --no-dev

# 非 root 运行：被攻破的 agent 也碰不到 root 拥有的文件。
RUN useradd --create-home --uid 1000 agent
USER agent

WORKDIR /workspace
ENTRYPOINT ["uv", "run", "--no-sync"]
CMD ["python", "-m", "pi_coding_agent"]
```

### 编排（`docker-compose.yml`，仓库根）

```yaml
# 把 agent 关进沙箱。
#
# 容器内 7 个工具以容器内当前用户权限执行。我们只挂载「单个」宿主目录作为
# 工作区，drop 全部 Linux capability + no-new-privileges 防提权。
# agent 仍需外联 LLM API，所以【不要】设 network: none。
services:
  pi-agent:
    build: .
    working_dir: /workspace
    volumes:
      - ./workspace:/workspace   # 只允许 agent 触碰这一个目录
    environment:
      PI_MOCK: "1"               # 或取消下面注释用真实 provider
      # PI_API: openai-completions
      # PI_PROVIDER: openrouter
      # OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    # 不要挂载 Docker socket 或任何宿主密钥。
    # read_only: true           # 若工作区只写 tmpfs，可开启进一步收敛
```

### 直接 `docker run`（不用 compose）

```bash
docker build -t pi-agent .

# mock 模式
docker run --rm -v "$PWD/workspace:/workspace" \
  --security-opt no-new-privileges --cap-drop ALL \
  pi-agent uv run --no-sync python -m pi_coding_agent

# 真实 provider（仅传所需那一把 key，不传 AWS/Docker 等其它凭证）
docker run --rm -v "$PWD/workspace:/workspace" \
  -e PI_API=openai-completions -e PI_PROVIDER=openrouter \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --security-opt no-new-privileges --cap-drop ALL \
  pi-agent uv run --no-sync python -m pi_coding_agent
```

### 隔离清单（checklist）

- [ ] 只挂载**一个**工作目录，绝不挂 `~/.ssh`、Docker socket、云凭证目录。
- [ ] 以**非 root** 用户运行（`USER agent` / `--user 1000:1000`）。
- [ ] `cap_drop: ALL` + `no-new-privileges` 防提权。
- [ ] **保留网络出口**（LLM 调用需要）；隔离的是文件系统/进程面，不是网络面。
- [ ] 跑真实 provider 时，只注入那一把 LLM key，不要顺手把宿主环境全部 `-e` 进容器。

## 8. 测试

```bash
uv run pytest        # 全量（当前 95 个，约 2~3s）
uv run pytest -q     # 静默
```

- 已启用 `pytest-timeout`（30s）+ `pytest-asyncio`（`asyncio_mode="auto"`）。
- 所有 provider / agent / 工具 / 遥测测试**均为 mock**，不触网、不卡死；个别挂起测试会在 30s 被护栏杀掉并打印 traceback。
- 想临时收紧超时调试：`uv run pytest <file> -o timeout=2`。

## 9. 端口对照（与原 TS 版）

| 原包 | 本包 | 备注 |
| --- | --- | --- |
| `ai` | `pi_ai` | 6 provider 全移植；`usage.input` 净去缓存读/写，与 OpenAI 引擎一致 |
| `agent` | `pi_agent_core` | `agent_loop` 双循环/工具调度/length 自愈复用 |
| `coding-agent` | `pi_coding_agent` | 7 工具 + 会话生命周期（/new /resume /fork /import）+ teardown |
| `tui` | `pi_tui` | 原版是自定义渲染引擎+原生绑定；本版用 **Textual** 达到"功能等价"，UI 允许不同，验收以事件流为准 |
| `telemetry` | `pi_telemetry` | 契约 + noop/memory/OTLP 适配 + 9 条一致性校验 |
