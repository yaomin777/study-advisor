# AI 经营学硕士留学申请顾问

基于 LangGraph 的多智能体聊天机器人，回答韩国经营学硕士项目的选校、申请要求、时间规划与文书问题。

> **免责声明**：本系统的建议仅供参考，请以学校最新招生简章为准；请勿输入真实个人信息（姓名、证件号、联系方式等）。

## 功能概览

- 9 个节点：router、profile_extractor、clarify、school_matcher、requirement_retriever、doc_reviewer、timeline_planner、verify、fallback。
- 学校数据在 `data/schools.json`，每条记录带官方出处 `source_url` 和整理日期 `as_of`。
- 申请要求与时间规划的回答由规则从数据生成，截止状态按当天日期计算；`verify` 核对日期和否定说法，回答找不到依据时转入 `fallback`。
- 两种运行模式：
  - **模型模式**：配置了 `ANTHROPIC_API_KEY` 时，以下节点调用 Claude（模型名见 `llm.py` 的 `MODEL_NAME`）：router（识别意图）、profile_extractor（抽取画像）、doc_reviewer（文书反馈）、school_matcher（只写每所学校的补充说明，符合度由代码判断）、verify（核对要求类陈述）。任何一次模型调用出错都会退回规则版。requirement_retriever 和 timeline_planner 始终由规则生成。
  - **演示模式（规则版）**：未配置密钥时，全部由规则完成，程序照常运行。
- 每个会话最多提 20 个问题。
- 侧边栏显示：运行模式、本轮节点路径、完整 trace（折叠区）、当前画像、本轮检索到的出处及数据整理日期。

## 本地运行

需要 Python 3.10 及以上（已在 3.13 测试）。

```bash
cd graph.py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 可选：配置密钥（不配置则为演示模式）
cp .env.example .env
# 编辑 .env，填写 ANTHROPIC_API_KEY=...

.venv/bin/streamlit run app.py
```

浏览器打开 http://localhost:8501 。密钥读取顺序：环境变量 → 本地 `.env` → `st.secrets`。

其他命令：

| 命令 | 作用 |
|---|---|
| `.venv/bin/python -m pytest -q` | 运行全部测试（使用假模型，不调用真实 API） |
| `.venv/bin/python smoke_live.py` | 真实调用模型的冒烟测试（需要密钥，会产生少量费用） |
| `.venv/bin/python export_graph.py` | 导出流程图 `graph.mmd`，并尝试导出 `graph.png`（需联网，失败时只保存 `graph.mmd`） |
| `.venv/bin/python graph.py` | 在命令行运行示例问题 |

## 部署（Streamlit Community Cloud）

1. 把项目推送到 GitHub。`.gitignore` 已排除 `.env` 和 `.streamlit/secrets.toml`，提交前请确认没有提交密钥。
2. 在 https://share.streamlit.io 选择 **Create app**，指定仓库、分支，主文件填 `app.py`；在 Advanced settings 中选择 Python 3.13。
3. 在应用的 **Secrets** 中填写：
   ```toml
   ANTHROPIC_API_KEY = "你的密钥"
   ```
   不填写时应用以演示模式运行。
4. 部署后，侧边栏的"运行模式"会显示当前是模型模式还是演示模式。

`.streamlit/config.toml` 已设置不在浏览器中显示错误堆栈。

## 测试场景

在界面中依次输入，检查回答和侧边栏：

| # | 输入 | 预期 |
|---|---|---|
| 1 | `我想选校，GPA 3.5`，然后 `选校：本科专业金融，雅思 7.0` | 第一轮追问语言成绩和本科专业（不追问 GMAT/GRE）；第二轮侧边栏画像合并两轮信息，回答逐校列出要求符合度和出处 |
| 2 | `成均馆大学需要 GMAT 吗？` | 回答"选择性提交……（请以最新简章为准）"；侧边栏出处显示官网链接和整理日期 2026-10-29 |
| 3 | `国民大学需要 GMAT 吗？` | 回答"官网未说明（核实日期 2026-10-29）（请以最新简章为准）"，不出现"不需要" |
| 4 | `首尔大学的申请要求是什么？` | 数据库中没有该校，节点路径为 `router → requirement_retriever → fallback`，回答建议查阅官网 |
| 5 | `帮我做一个国民大学的申请时间规划` | 列出申请轮次，按当天日期标注"未截止（距截止 N 天）"或"已截止（以最新简章为准）" |

## 当前限制

- 学校数据目前只有 5 所。
- 选校不做冲刺/匹配/保底分档：school_matcher 逐校列出语言、本科专业、GMAT/GRE、工作经验、截止状态的符合度（符合 / 不符合 / 无法判断 / 官网未说明），由代码根据学校记录判断；学校记录没有 GPA 要求，因此不判断 GPA。英语成绩在记录没有具体分数时一律为"无法判断，以简章为准"。
- 对话不跨会话保存；刷新页面即开始新会话。
