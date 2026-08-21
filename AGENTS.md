<project>
源码路径：E:\ftre-agent-core\src\ftre_agent_core\
后端路径：E:\ftre\src\ftre\（引用本库）
定位：Agent 核心库（无状态、纯算法）被 ftre 后端 import 使用，不独立部署
技术栈：Python 3.12
日志：logging

MANDATORY 首次进入本仓库先读 3 份文档，之后每次 commit 前重读第 1 份：
1. docs/COMMIT.md — 提交规范唯一完整定义（type/scope/hook 机制）
2. docs/PROCESS.md — PRD 驱动开发流程（六步闭环）
3. docs/TODO.yaml — 阶段 id 唯一事实源（commit scope 校验依据）
</project>

<git_flow MANDATORY>

<basic_discipline>
- NEVER 私自 commit / push：除非用户明确要求（"commit"、"push"、"提交"），否则只改代码不提交
- 回滚需确认：回滚前告知内容/范围/影响，得到确认后再执行
- ALWAYS push 前先 commit；多仓库联动（改 core 后同步验证 ftre 后端）
- 跨仓库操作必须 set_workspace 显式切换：`cd A && git ...` 中的 cd 不改变 bash 工具工作区
</basic_discipline>

<branch_model>
master（仅发布，永不直接提交）← develop（默认基底，只接受 PR 合入）← feature/&lt;阶段id&gt;-&lt;name&gt; / prd-update / todos-update / release/&lt;ver&gt; / hotfix/&lt;name&gt;

- 默认工作分支是 develop
- NEVER 直接提交 master；NEVER 直接 commit 到 develop——**全 PR 流：develop 只接受 GitHub PR 服务器端合入，本地 develop 永远只 pull 同步**（pre-push hook 强制）
- MANDATORY feat/fix 分支名必须关联 TODO 阶段 id（如 feature/A2-tool-system），提交 scope 与分支名阶段 id 必须一致（commit-msg hook 强制）
</branch_model>

<commit_format>
`&lt;type&gt;(&lt;scope&gt;): &lt;subject&gt;`，subject 中文
- type 白名单：feat / fix / prd / todos / docs / refactor / test / style / chore / perf
- feat/fix/prd/todos 的 scope 必须是 docs/TODO.yaml 中真实存在的阶段 id
- feat 额外强制：暂存必须包含对应阶段 PRD（docs/prd/PRD-&lt;scope&gt;-*.md）——行为变更必须同步 PRD 变更记录（无 PRD 的基建阶段跳过）
- perf 的 scope 必须带 FR 引用（如 perf(C2-FR6)），引用的 FR 编号必须真实存在于对应 PRD
- 其他 type 的 scope 用模块名（check_commit_msg.py 顶部「裁剪点」MODULE_SCOPES）
- 一条提交只做一件事；NEVER 写 fix stuff / update / misc 这类无意义 message
</commit_format>

<merge_and_hooks>
- feature/* → develop 一律走 GitHub PR/MR（Code Review）：push 分支后提 PR，NEVER 本地 `git merge --no-ff` 合并回 develop
- develop 与 master 之间同样禁止本地 merge，一律走 PR：develop → master 走 release/* 提 PR；hotfix 回灌走 PR
- NEVER rebase 已推送历史
- 本地强制：.githooks/commit-msg（提交校验）+ .githooks/pre-push（master 双保护 + develop 三重保护：禁删 / 禁 feature 直推 / 本地领先即拒）
- merge:/revert: 开头系统提交跳过
- MANDATORY 首次在本仓库提交前，先完整阅读 docs/COMMIT.md（提交规范唯一完整定义，含 type/scope 规则与常见错误速查）
- 标准流程：checkout develop && pull → checkout -b feature/&lt;阶段id&gt;-&lt;task&gt; → 开发+测试 → commit → push origin feature/&lt;task&gt; → GitHub 提 PR 合入 develop → checkout develop && pull 同步
</merge_and_hooks>

</git_flow>

<prd_driven MANDATORY>
- MANDATORY 首次在本仓库开工前，先完整阅读 docs/PROCESS.md（PRD 驱动流程六步闭环）
- ALWAYS 先 PRD 后开发：TODO 阶段开工前先在 docs/prd/ 建 PRD（从 PRD-TEMPLATE.md 复制）并定稿 approved
- PRD 是唯一依据：需求/实现/测试/验收全部对照 PRD；验收按 PRD「验收标准」逐条核对
- 阶段 id 与状态见 docs/TODO.yaml（commit scope 的唯一事实源）
</prd_driven>

<architecture>

<modules>
- agent/：ReActAgent（react.py，ReAct 循环）+ runner/
- llm/：LLM 调用（LLMAdapter、StreamChunk、协议适配器与 registry 工厂）
- tool/：工具体系（base.py Tool/ToolParameter/Injected / registry.py ToolRegistry / cancellation.py）
- hooks.py：宿主注入的 HookDispatcher、HookSpec 与 Tool/LLM/turn-stopping typed contracts
- message/：Msg/MsgName + ContentBlock（text/toolCall/toolResult/image/think）
- event/：EventBase/CustomEvent（被 Tracer 捕获）
- permission/：权限引擎（allow/deny/ask，_engine.py/_types.py/_context.py）
- state/：AgentState（_agent_state.py）
- threading.py：线程安全工具
- tracing.py：Tracer 调用追踪 + SQLite 导出
</modules>

<core_invariants>
- 无状态：本库不持有进程级可变状态，不依赖具体 Channel/后端
- ToolHandler.run_one() 返回结构化 ToolResult；原始工具的 str / EventBase / tuple[str, dict] 在 Core 内归一化
- Hook Dispatcher 为 async protocol；宿主负责 await、Scope、生命周期和 failure policy
</core_invariants>

</architecture>

<run_and_test>
- 测试：pytest（tests/ 目录，镜像包结构）
- 代码规范：ruff（black 风格）+ isort
- MANDATORY 提交/合并前通过：`pytest` + `ruff check .`
- 测试不依赖真实 API key——用 mock/fake provider
- 新功能必须配测试；bug 修复必须配回归测试
</run_and_test>

<anti_lazy>
- NEVER 用空函数、TODO、placeholder 假装完成
- NEVER 重复性任务做几个就声称全部完成——逐个执行，验证全部
- NEVER 跳过失败的步骤——修复后重新验证
- 同一问题反复改不好就停下：回到初始假设、复现路径和失败证据重新判断，换方向
- 收尾前通读改过的文件：确认连贯、无语法错误、无残留调试代码
- 违反以上任何一条：下一轮立即自纠
</anti_lazy>
