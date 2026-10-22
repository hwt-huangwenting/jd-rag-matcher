# 实现 Prompt：JD 匹配问答器（RAG 小型实战）

> 用法：把本文件作为项目唯一的实现说明交给编码 Agent（或作为我们后续分阶段实现的依据）。
> 要求按「里程碑」逐步交付，每完成一个阶段先自测并汇报，确认后再进入下一阶段。

---

## 一、角色与目标

你是一名 Python 后端工程师。请用**纯 Python**实现一个命令行 RAG 应用：把一份脱敏简历（Markdown）和若干招聘 JD（txt/md）作为知识库，用户输入任意问题（例如「这个岗位要求 RAG 经验，我匹配吗？」），程序检索最相关的文本块，调用大模型生成**带引用来源**的回答。

这是一个用于展示 RAG 底层理解的练手项目，**可读性、正确性和工程完整度优先于功能数量**。

---

## 二、硬性约束（不得违反）

1. **运行环境**：Python 3.11+。
2. **第三方依赖仅限**：`requests`、`numpy`、`python-dotenv`。不得引入其他 pip 包。
3. **禁止使用任何 RAG 框架或向量数据库**：包括但不限于 LangChain、LlamaIndex、Chroma、FAISS、Milvus、Qdrant、pgvector、sentence-transformers。向量持久化用 JSON 文件，余弦相似度用 `numpy` 手写，检索用 `top-k`。
4. **模型**（智谱开放平台，OpenAI 兼容协议）：
   - Base URL：`https://open.bigmodel.cn/api/paas/v4`
   - 对话模型：`glm-4-flash`
   - Embedding 模型：`embedding-3`，向量维度固定 **1024**（请求时显式传 `dimensions: 1024`，加载索引时校验维度）
   - 鉴权方式：`Authorization: Bearer <API_KEY>`
5. **密钥管理**：API Key 只能从 `.env` 读取，不得硬编码、不得写入日志。`.env` 必须被 `.gitignore` 排除；仓库只提交 `.env.example`。
6. **隐私**：代码、文件名、注释、知识库中**不得出现真实姓名、学号、手机号、邮箱、身份证号、住址、公司名**。简历可基于真实经历，但必须脱敏：姓名等身份信息删除，学校统一写「某高校」，项目名与量化结果可保留；JD 删去公司名与联系方式。简历 PDF 原件与 `private/` 目录不得提交，`*.pdf`、`private/` 加入 `.gitignore`。
7. **不覆盖、不删除**用户已有的非本项目文件。

---

## 三、功能需求

### 3.1 知识库

- 目录：`knowledge/`，支持 `.md` 和 `.txt`。
- 至少包含：
  - `knowledge/resume.md`：真实经历脱敏后的简历（教育背景、技能、项目经历）。
  - `knowledge/jd/` 下 3 份真实 AI 应用开发方向 JD：`jd_ai_app.md`、`jd_ai_intern.md`、`jd_ai_agent.md`，每份带「岗位名称」字段用于区分。
- 每个文档必须能追溯到**文件名 + 标题/章节路径**，用于回答中的引用标注。

### 3.2 切块（chunking）

- Markdown：按 ATX 标题（`#`/`##`/`###`）切分章节；章节正文若超过 `CHUNK_SIZE`，按句子边界继续切分，保留 `CHUNK_OVERLAP` 字符重叠。
- TXT：按空行切段落，短段落合并到接近 `CHUNK_SIZE`，超长段落按句子切分。
- 默认值：`CHUNK_SIZE=500`（字符），`CHUNK_OVERLAP=80`，均可在 `.env` 覆盖。
- 每个 chunk 的元数据至少包含：
  ```json
  {
    "id": "sha1前12位",
    "source": "knowledge/jd/jd_ai_app.md",
    "section": "岗位要求 > 技术栈",
    "text": "……",
    "char_count": 320
  }
  ```
- `id` 由 `source + section + 序号` 计算稳定哈希，保证重复建索引时 ID 不变。
- 切块结果需支持 `--show-chunks` 打印，便于人工检查。

### 3.3 建索引

- 命令：`python main.py build [--force]`
- 流程：加载文档 → 切块 → 批量调用 `embedding-3` → 写入 `data/index.json`。
- 批量请求：每批默认 16 条，可配置；批间做基本限速与失败重试（指数退避，最多 3 次）。
- 索引文件结构：
  ```json
  {
    "model": "embedding-3",
    "dim": 1024,
    "created_at": "ISO8601",
    "chunks": [
      {"id": "...", "source": "...", "section": "...", "text": "...", "char_count": 320, "vector": [0.01, ...]}
    ]
  }
  ```
- 若 `data/index.json` 已存在且 `--force` 未指定，提示用户确认是否覆盖；维度或模型不一致时直接拒绝加载并给出明确错误。
- 建索引完成后打印：文档数、chunk 数、总耗时、索引入口路径。

### 3.4 检索

- 命令：`python main.py ask "问题" [--top-k 6] [--min-score 0.52] [--show-context]`
- 对问题调用同一 Embedding 模型得到查询向量。
- **手写余弦相似度**：`dot(q, d) / (norm(q) * norm(d) + 1e-8)`，用 `numpy` 向量化计算全部 chunk 的得分，不得调用 sklearn/scipy 的相似度函数。
- 按得分降序取 `top-k`，并过滤低于 `min-score` 的结果。
- `--show-context` 输出每个命中 chunk 的得分、来源、章节和正文摘要，便于调试。
- 若没有任何结果超过阈值：**不调用大模型**，直接返回「知识库中没有找到足够相关的依据」并结束。
- 默认阈值 `MIN_SCORE=0.52`。智谱 `embedding-3` 对中文文本的余弦分数整体偏高，0.3 几乎不会触发拒答；阈值与语料、模型相关，**必须实测校准**：用 3 个明显相关的问题和 3 个明显无关的问题跑 `--show-context`，记录得分区间后再微调。当前知识库实测：无关问题最高 0.4904，评测所需最低相关块 0.5673，取 0.52；`TOP_K=6`，因为一次匹配回答需要同时容纳简历技能与多份 JD 的要求。
- 校准必须使用**真实提问方式**（如「这个岗位要求 RAG 经验，我匹配吗」），不要只拿 JD 原文当查询，否则分数会虚高。

### 3.5 生成回答（带引用）

- 将检索到的片段编号为 `[1]...[k]`，连同来源信息一起拼进 prompt。
- System Prompt 必须约束：
  1. 只依据提供的资料回答，不得编造；
  2. 每条关键结论后标注引用编号，如 `[1]`；
  3. 资料不足时明确说明「知识库中没有相关依据」，不要猜测；
  4. 涉及「是否匹配」时，分别列出「岗位要求」与「候选人证据」，再给出结论。
- 对话参数：`temperature=0.2`，`max_tokens` 可配置（默认 800）。
- CLI 输出格式：
  ```
  回答
  ----
  ……（正文，含 [1][2] 标注）

  引用来源
  ----
  [2] knowledge/resume.md · 专业技能（相似度 0.6087）
  [4] knowledge/jd/jd_ai_app.md · 任职要求（相似度 0.6038）
  ```
- 引用来源必须由检索结果映射生成，**不得让模型自己编来源**。
- 生成阶段做一次兜底：资料存在但回答既没有 `[n]` 也不属于拒答时，带上提醒重试一次；重试仍无引用则保留原回答。

### 3.6 交互模式与网页界面（可选，最后做）

- `python main.py chat` 进入多轮问答；每轮独立检索（不实现复杂记忆），输入 `exit` 退出。
- `python web.py` 启动本地网页（默认 `http://127.0.0.1:8000`），提供提问、引用卡片、召回块与相似度展示。
- 网页提供「知识库管理」：上传 `.md` / `.txt` 简历与 JD，保存后自动重建索引；仅本机可写，单文件上限 200 KB。
- 网页只使用 Python 标准库 `http.server` + 原生 HTML/CSS/JS，不新增 pip 依赖；API Key 只留在服务端。
- 不做用户系统、数据库、登录与多会话隔离，默认只监听本机，不暴露公网。

### 3.7 配置

`.env.example` 至少包含：

```dotenv
ZHIPU_API_KEY=your_api_key_here
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
CHAT_MODEL=glm-4-flash
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
EMBED_BATCH_SIZE=16
TOP_K=6
MIN_SCORE=0.52
CHUNK_SIZE=500
CHUNK_OVERLAP=80
MAX_TOKENS=800
REQUEST_TIMEOUT=60
```

- 缺少 `ZHIPU_API_KEY` 时，在启动阶段给出人类可读的错误提示（指向 `.env.example`），不要抛裸异常。
- 所有路径基于项目根目录解析，不依赖当前工作目录。

### 3.8 确定性评测（eval.py）

- 新增 `eval.py` 与 `eval_cases.json`，用 Python 标准库实现，**不引入 pytest 等测试框架**。
- 用例共 10 条：8 条可回答 + 2 条应拒答。每条只包含：问题、类型、期望来源文件（不写标准答案，避免评测退化成模型输出比对）。
- 判定规则保持确定性，不评价文风：
  - 可回答：回答至少含 1 个 `[n]`；所有 `[n]` 都能在本次检索结果中找到；引用的来源文件中包含期望来源文件。
  - 应拒答：检索为空，或回答明确说明知识库无依据，且不产生引用。
- 输出每条用例的通过/失败及失败原因，最后打印 `通过数/总数` 与通过率。
- 失败原因至少能区分：检索未命中、引用编号越界、引用来源错误、该拒答未拒答。
- 评测会真实调用 embedding 与对话接口；失败用例需人工复核，不得为了刷分修改判定逻辑。
- 评测脚本不得依赖交互输入，可直接 `python eval.py` 运行。

---

## 四、目录结构（建议）

```
jd-rag-matcher/
├── main.py                 # CLI 入口（argparse）
├── web.py                  # 零依赖本地 Web 服务（http.server）
├── eval.py                 # 10 条用例的确定性评测
├── eval_cases.json         # 评测用例（问题 / 类型 / 期望来源）
├── jdqa/
│   ├── __init__.py
│   ├── config.py           # .env 加载、默认值与校验
│   ├── loader.py           # 文档读取
│   ├── chunker.py          # Markdown / TXT 切块
│   ├── embedder.py         # embedding-3 批量调用 + 重试
│   ├── store.py            # index.json 读写与校验
│   ├── retriever.py        # 手写余弦 + top-k
│   ├── generator.py        # glm-4-flash 调用与引用拼装
│   └── cli.py              # build / ask / chat 子命令
├── knowledge/
│   ├── resume.md
│   └── jd/
│       ├── jd_ai_app.md
│       ├── jd_ai_intern.md
│       └── jd_ai_agent.md
├── docs/                   # README 演示用真实运行输出图
│   ├── demo.svg
│   └── web.png
├── web/
│   └── index.html          # 单页界面（原生 HTML/CSS/JS）
├── data/                   # 运行时生成 index.json（gitignore）
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

> 允许小幅调整结构，但不要为「将来可能的需求」增加抽象层。每个模块只做一件事。

---

## 五、实现里程碑（必须按顺序，逐个验收）

### M1 项目骨架与配置
- 创建目录、`requirements.txt`、`.gitignore`、`.env.example`、`config.py`、`main.py` 空壳。
- `.gitignore` 至少忽略：`.env`、`data/`、`__pycache__/`、`*.pyc`、`.venv/`、`.idea/`。
- **验收**：`python main.py --help` 正常输出；缺少 Key 时能给出友好提示；打印配置时 Key 被掩码。

### M2 知识库与切块
- 写入 1 份真实经历脱敏简历 + 3 份真实 JD（去掉公司名与联系方式）。
- 实现 `loader.py`、`chunker.py`，提供 `python main.py build --dry-run --show-chunks` 仅打印切块结果。
- **验收**：每个 chunk 有 source/section/text；Markdown 标题路径正确；无个人信息；chunk 长度不超过上限（允许少量因句子边界溢出，需说明）。

### M3 Embedding 与索引落盘
- 实现 `embedder.py`、`store.py`、`build` 正式流程。
- **验收**：生成 `data/index.json`，`dim=1024`，`len(vector)==1024`，chunk 数与 dry-run 一致；重复执行不产生重复条目（覆盖写）。

### M4 检索
- 实现 `retriever.py` 与 `ask --show-context`（此阶段先不接生成，只打印检索结果）。
- 用 3 个明显相关、3 个明显无关的真实问法做阈值校准，记录得分区间。
- **验收**：问题「这个岗位要求 RAG 经验，我匹配吗」的 Top-3 同时命中 JD 技能段与简历项目段；余弦得分在 `[-1, 1]` 且随相似度合理排序；无关问题在 `MIN_SCORE=0.5` 下返回空结果；校准得到的实际得分区间与最终阈值记入 README。

### M5 生成与引用
- 实现 `generator.py`，接通 `glm-4-flash`，输出答案与引用来源。
- **验收**：
  - 回答中的 `[n]` 均能在引用列表中找到对应项；
  - 提问知识库之外的内容时，明确说明无依据，不编造；
  - 网络/鉴权错误给出可操作的错误信息，不泄露 Key。

### M6 确定性评测与健壮性
- 实现 `eval.py` 与 `eval_cases.json`（8 条可回答 + 2 条应拒答）。
- 增加请求超时、重试、空输入校验、索引缺失提示、`chat` 模式（可选）。
- **验收**：`python eval.py` 输出每条用例结果与通过率；失败用例能定位到「检索未命中 / 引用编号越界 / 引用来源错误 / 该拒答未拒答」中的具体原因；无新增依赖。

### M7 README 与演示
- 写 `README.md`：安装、配置、构建索引、提问示例、评测方法、目录说明，并记录 3 个设计取舍（切块策略、为何手写余弦、引用如何保证可追溯）。
- README 顶部放 `docs/demo.gif`（或 `demo.png`）：必须是真实运行的「提问 → 带引用回答」录屏或截图，同时保留文本版输出作为兜底。
- **验收**：按 README 从零执行一遍可跑通；README 首屏演示图能正常显示；无新增依赖；代码中无 TODO 遗留；仓库无个人信息。

---

## 六、代码与交付要求

1. 使用类型注解；公共函数写简洁 docstring；注释解释「为什么」而非「做了什么」。
2. 网络请求统一封装：设置超时、捕获 HTTP 错误、解析错误响应体，重试仅针对 429/5xx/超时。
3. 日志或输出中不得打印 API Key、完整请求头、完整向量。
4. 不要一次性写完全部里程碑；**每阶段先说明要改哪些文件，再动手**。
5. 每阶段结束输出：
   - 变更文件清单
   - 自测命令与关键输出
   - 已知限制
   然后等待确认再继续。
6. 不添加未被要求的功能（Web UI、缓存系统、并发、Docker 等一律不做）；不引入 pytest 等测试框架，评测逻辑用标准库自写在 `eval.py` 中。

---

## 七、最终验收清单

- [ ] `pip install -r requirements.txt` 仅安装 3 个依赖
- [ ] `python main.py build` 成功生成 `data/index.json`
- [ ] `python main.py ask "这个岗位要求 RAG 经验，我匹配吗"` 返回带 `[1][2]` 引用的回答
- [ ] 引用来源可定位到具体文件与章节
- [ ] 无关问题返回「知识库无依据」，不编造
- [ ] `data/index.json` 维度为 1024，向量字段长度一致
- [ ] `python eval.py` 可运行，输出 10 条用例结果与通过率
- [ ] README 记录了阈值校准的实际得分区间与最终取值
- [ ] README 首屏有真实运行的演示截图或 GIF，且能正常显示
- [ ] 仓库中无 `.env`、无 `data/`、无姓名学号等个人信息
- [ ] README 可按步骤复现
