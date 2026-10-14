# Claude Code tool search (deferred tools)

## English

An MCP server with many tools can fill the context before the conversation
starts. rea-agents 6.4.0 registers 170 tools, about 140K tokens of
definitions, so a Claude Code request with it went from 133K to 272K tokens
and was rejected against the 200K window.

Claude Code solves this with tool search. Rarely used tools are sent with
`defer_loading: true`, their names are listed in a system reminder, and the
model loads one by calling the `ToolSearch` tool, whose result carries a
`{"type": "tool_reference", "tool_name": ...}` block. The Anthropic API
expands that block into the tool definition. Flyto MLX now does the same:

- Tools with `defer_loading: true` are not rendered into the chat template's
  tool list, but the tool-call parser still accepts them.
- A `tool_reference` inside a `tool_result` becomes
  `<functions>\n<function>{"description", "name", "parameters"}</function>\n</functions>`,
  the shape Claude Code's ToolSearch description promises the model.
- Loading a tool never changes the prompt head, so the prefix cache stays
  valid. A loaded tool appears in later requests' `tools` with
  `defer_loading: true`, so the template still skips it.

### Client settings

Claude Code 2.1.296 keeps tool search off for a non-Anthropic base URL. Both
of these are needed in the `env` block of `~/.claude/settings.json`:

- `"ENABLE_TOOL_SEARCH": "true"`
- remove `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS`. While it is set, the
  client forces tool search off whatever `ENABLE_TOOL_SEARCH` says (only a
  managed-settings `force` overrides it).

Turning the experimental betas on also adds `thinking.display: "omitted"` and
`context_management: {"edits": [{"type": "clear_thinking_20251015", "keep":
"all"}]}` to requests. The server ignores both. The second is a no-op anyway.

Do not add `mid_conv_tool_change` to `CLAUDE_CODE_MODEL_CAPABILITIES` for
local models. That capability changes how the client reports tool changes
mid-conversation, and only the shape captured above (loaded tools appended to
`tools` with `defer_loading`) has been verified against this server.

Check the decision with `claude -p hi --debug`. The debug log should show
`[ToolSearch:optimistic] mode=tst, ENABLE_TOOL_SEARCH=true, result=true`.
With tool search on, a request with rea connected went from 540 KB to 51 KB.

---

## 中文

MCP 服务工具一多, 对话还没开始上下文就被占满. rea-agents 6.4.0 注册了 170 个工具,
定义约 14 万 token, 接上之后 Claude Code 的请求从 13.3 万涨到 27.2 万, 超过 20 万窗口被拒.

Claude Code 用"工具按需加载" (tool search) 解决这个问题. 不常用的工具带
`defer_loading: true` 发送, 名字写在一条 system reminder 里, 模型要用时调
`ToolSearch`, 它的结果里是 `{"type": "tool_reference", "tool_name": ...}`,
由 Anthropic API 把它展开成工具定义. Flyto MLX 现在做同样的事:

- 带 `defer_loading: true` 的工具不进模板的工具列表, 但工具调用解析器仍然认它们.
- `tool_result` 里的 `tool_reference` 换成
  `<functions>\n<function>{"description", "name", "parameters"}</function>\n</functions>`,
  这正是 Claude Code 的 ToolSearch 工具描述里向模型承诺的格式.
- 加载工具不会改动提示开头, 前缀缓存保持有效. 加载过的工具在之后的请求里仍带
  `defer_loading: true` 出现在 `tools` 中, 模板照样跳过.

### 客户端设置

Claude Code 2.1.296 对非 Anthropic 的 base URL 默认不开 tool search.
`~/.claude/settings.json` 的 `env` 里两件事都要做:

- 加 `"ENABLE_TOOL_SEARCH": "true"`
- 去掉 `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS`. 只要它在, 客户端就强制关掉 tool search,
  不管 `ENABLE_TOOL_SEARCH` 怎么设 (只有托管设置里的 `force` 能压过它).

打开实验特性后请求还会多出 `thinking.display: "omitted"` 和
`context_management: {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}`,
服务端都忽略, 后者本来也是空操作.

不要给本地模型在 `CLAUDE_CODE_MODEL_CAPABILITIES` 里加 `mid_conv_tool_change`.
这个能力会改变客户端在对话中途通报工具变化的方式, 目前只验证过上面抓到的形状
(加载过的工具带 `defer_loading` 追加到 `tools`).

用 `claude -p hi --debug` 核对, 调试日志应出现
`[ToolSearch:optimistic] mode=tst, ENABLE_TOOL_SEARCH=true, result=true`.
开启后, 接着 rea 的请求从 540 KB 降到 51 KB.
