# STT aligner companion misconfiguration (word timestamps / energy_tripass return empty)

Status: root cause confirmed empirically on m5max, 2026-07-12. Fix is a
one-model configuration change, not a code change. The transcription
pipeline, the admin aligner config panel, and the client docs already exist
and are correct.

## Symptom

Calling `POST /v1/audio/transcriptions` on the stereo sales-call model with
`word_timestamps=true`:

- mono clip: transcript text comes back, but `segments[0].words` is empty
  (0 words). Length-independent.
- stereo clip with `diarize_backend=energy_tripass` + `left_speaker` /
  `right_speaker`: the response is completely empty -- `text=""`,
  `segments=[]`, no speaker attribution.

The forced aligner (`qwen3-forced-aligner-0.6b-4bit`) works fine when called
directly (`model=<aligner>` + `text=<transcript>` returns per-word times).

## Root cause

`_maybe_align_inplace` in `omlx/api/audio_routes.py` reads the ASR model's
`aligner_model` via `settings_manager.get_settings(resolved_model)`.
`resolved_model` is the registry entry id -- the lowercase
`qwen3-asr-1.7b-audio8-text4` (this is also the id the server advertises in
`/v1/models` and the id clients send).

But in `~/.fmlx/model_settings.json` the aligner is configured under a
different key -- the canonical HF name `Qwen3-ASR-1.7B-audio8-text4-mlx`.

`get_settings` is a plain dict lookup with no normalization, and
`resolve_model_id` has no link between the two names (the entry id resolves to
itself; the canonical name resolves to nothing). So:

- `get_settings("qwen3-asr-1.7b-audio8-text4")` -> default settings,
  `aligner_model=None` -> aligner companion silently skipped -> 0 words.
- `get_settings("Qwen3-ASR-1.7B-audio8-text4-mlx")` -> `aligner_model=
  Qwen3-ForcedAligner-0.6B-4bit`.

Proven with the same code + the same on-disk settings file the live server
uses (a `ModelSettingsManager` loaded from `~/.fmlx`).

The stereo `energy_tripass` total-empty result is a downstream symptom of the
same bug, not a second bug: `tripass_merge` is entirely word-driven
(`_flatten_words` reads `segments[*].words` from each of the L / R / mix
passes). With the aligner skipped every pass has 0 words, so the merge
produces 0 segments. Once the aligner fires, the merge works -- verified by an
offline end-to-end simulation on a real 60s stereo segment: L 114 / R 82 /
mix 176 aligned words -> 19 speaker turns, 97 words on the sales side, 85 on
the customer side, coherent back-and-forth with word-level timestamps.

## Fix

Set the forced aligner for the ASR model under its current entry id. The
admin UI already has the control:

Admin dashboard -> model `qwen3-asr-1.7b-audio8-text4` -> Model Settings ->
Forced aligner -> select `qwen3-forced-aligner-0.6b-4bit` -> Save.

`set_settings` updates the in-memory settings map immediately, so this takes
effect on the next request with no reload and no server restart. The write is
keyed by the entry id (the admin PUT path uses the entry id), so it lands on
the key `get_settings` actually reads.

Optionally set `aligner_max_audio_seconds` (default 270s) in the same panel;
a full 480s call needs `on_aligner_overflow=chunk` to align past the limit.

The stale `Qwen3-ASR-1.7B-audio8-text4-mlx` entry can be left in place (it is
dead weight -- no entry matches it) or removed during cleanup.

Only m5max serves this model. m2max's local server has no audio models, so
there is nothing to apply there (a PUT would 404 -- the entry does not exist).

## Applied + verified live (2026-07-12, m5max)

Applied via the admin API (`auto-login?key=<main api key>` for a session
cookie, then `PUT /admin/api/models/qwen3-asr-1.7b-audio8-text4/settings` with
`{"aligner_model": "qwen3-forced-aligner-0.6b-4bit"}`). `active_requests` was
0 at the time. The write takes effect live; no reload, no restart.

Note the aligner value must be the entry id `qwen3-forced-aligner-0.6b-4bit`,
not the canonical `Qwen3-ForcedAligner-0.6B-4bit` -- the latter does not
resolve (case-insensitive match fails on the `forcedaligner` vs
`forced-aligner` hyphenation), so the old stale entry was doubly broken (wrong
outer key AND an unresolvable aligner value).

Live results on the running server:

- mono 60s clip, `word_timestamps=true`: 138 chars, 114 word timestamps
  (was 0 before the fix).
- stereo 60s clip, `energy_tripass` + `left_speaker=销售` /
  `right_speaker=客户`: 19 speaker turns, balanced sales/customer, per-word
  times (was completely empty before the fix).
- full 480s stereo call, `energy_tripass` + `on_aligner_overflow=chunk`:
  108 turns, 1721 words, 54 sales / 54 customer turns, coverage to 479.1s,
  coherent open-to-close dialogue. ~76s wall-clock for the 3-pass + chunked
  alignment.

## Related risk: settings-key drift (note, not fixed here)

This is one instance of a systemic pattern: 20 of 27 settings keys in
m5max's `model_settings.json` do not match any current entry id. Models were
renamed / re-registered over time and their old settings entries were left
orphaned under the old names. Any such model silently loses ALL its per-model
settings (context window, thinking, dflash, aligner, ...), because every
`resolve_model_id` -> `get_settings` consumer looks up by entry id.

Do NOT fix this with fuzzy matching in `get_settings` -- the drifted keys are
near-duplicates (`gemma4-e2b` / `gemma4-e2b-4bit`, `qwen-dense-9b` /
`qwen-dense-4b`), and any strip-suffix / case-insensitive fallback risks
applying a stale config to the wrong model. A safe cleanup is a separate,
explicit audit: for each orphaned key, either re-key it to the matching entry
id or delete it, one at a time, with a human in the loop.

Audit outcome (2026-07-12, m5max): all 20 orphaned keys carry meaningful
non-default settings (dflash / specprefill draft-model paths, reasoning
parsers, thinking budgets, sampling, mtp, model_type_override) -- none are
all-default dead weight. Only ONE had an unambiguous 1:1 current-entry match
(`Qwen3-ASR-...-mlx` -> `qwen3-asr-1.7b-audio8-text4`), and that one is the
aligner fix above. The other 19 have NO exact current-entry match; fuzzy
candidates are ambiguous (e.g. `gemma4-moe-26b-a4b` plausibly maps to the q4,
q6 OR q8 quant -- applying an old dflash draft to the wrong quant could crash
or degrade), so they need the rename history that only the owner has. Some
dflash draft-model dirs referenced by these keys still exist on disk
(gemma4-dense-31b-dflash-draft, gemma4-moe-26b-a4b-dflash-draft) while others
are gone (gemma4-26b-a4b-draft, gemma4-moe-26b-a4b-assistant). Recommended
process: owner confirms each old-key -> current-entry mapping (or "delete"),
one row at a time; then re-key via the admin PUT settings path so the write
lands on the entry id. Not auto-applied.

## Cosmetic (unrelated)

The `duration` field returns a tiny residual value (~0.3s) instead of the
audio duration, on good transcriptions too. Independent of the aligner; not
addressed here.

---

# STT 对齐 companion 配置错位 (词级时间戳 / energy_tripass 返回空)

状态: 2026-07-12 在 m5max 上实证确认根因。修复是**单个模型的配置变更, 不是代码
改动**。转写流水线, admin 对齐配置面板, 客户端文档都已存在且正确。

## 现象

对立体声电销模型调 `POST /v1/audio/transcriptions` 且 `word_timestamps=true`:

- 单声道: 转写文本正常返回, 但 `segments[0].words` 为空 (0 词)。与音频长度无关。
- 立体声 + `diarize_backend=energy_tripass` + `left_speaker` / `right_speaker`:
  响应完全为空 -- `text=""`, `segments=[]`, 无说话人归属。

而 forced aligner (`qwen3-forced-aligner-0.6b-4bit`) 直连调用完全正常
(`model=<aligner>` + `text=<转写文本>` 出词级时间戳)。

## 根因

`omlx/api/audio_routes.py` 的 `_maybe_align_inplace` 通过
`settings_manager.get_settings(resolved_model)` 读 ASR 模型的 `aligner_model`。
`resolved_model` 是注册表 entry id -- 小写的 `qwen3-asr-1.7b-audio8-text4` (也是
server 在 `/v1/models` 里暴露, 客户端实际发送的 id)。

但在 `~/.fmlx/model_settings.json` 里, aligner 配置存在**另一个 key** 下 -- 规范
HF 名 `Qwen3-ASR-1.7B-audio8-text4-mlx`。

`get_settings` 是不做归一化的纯 dict 查找, 且 `resolve_model_id` 在这两个名字之间
没有任何链接 (entry id 解析到自身; 规范名解析不到任何东西)。所以:

- `get_settings("qwen3-asr-1.7b-audio8-text4")` -> 默认 settings,
  `aligner_model=None` -> 对齐 companion 被静默跳过 -> 0 词。
- `get_settings("Qwen3-ASR-1.7B-audio8-text4-mlx")` -> `aligner_model=
  Qwen3-ForcedAligner-0.6B-4bit`。

用运行中 server 使用的同一份代码 + 同一份磁盘 settings 文件证实 (从 `~/.fmlx` 加载
一个 `ModelSettingsManager`)。

立体声 `energy_tripass` 整体返回空是同一 bug 的**下游症状, 不是第二个 bug**:
`tripass_merge` 完全由词驱动 (`_flatten_words` 从 L / R / mix 三个 pass 的
`segments[*].words` 取词)。对齐被跳过后每个 pass 都 0 词, 合并出 0 段。一旦
aligner 触发, 合并就正常 -- 用一段真实 60s 立体声做离线端到端模拟验证: L 114 /
R 82 / mix 176 个对齐词 -> 19 个说话人轮次, 销售侧 97 词, 客户侧 85 词, 连贯的
一问一答 + 词级时间戳。

## 修复

在 ASR 模型的当前 entry id 下设置 forced aligner。admin UI 已有该控件:

Admin 面板 -> 模型 `qwen3-asr-1.7b-audio8-text4` -> Model Settings ->
Forced aligner -> 选 `qwen3-forced-aligner-0.6b-4bit` -> 保存。

`set_settings` 立即更新内存中的 settings map, 所以下一个请求即生效, 无需 reload,
无需重启 server。写入按 entry id 归 key (admin PUT 路径用 entry id), 正好落在
`get_settings` 实际读取的那个 key 上。

可在同一面板设 `aligner_max_audio_seconds` (默认 270s); 完整 480s 通话需带
`on_aligner_overflow=chunk` 才能对齐超限部分。

历史遗留的 `Qwen3-ASR-1.7B-audio8-text4-mlx` 条目可以留着 (它是死配置 -- 没有任何
entry 匹配它) 或清理时删掉。

只有 m5max serve 这个模型。m2max 本地 server 没有任何 audio 模型, 无需在那边应用
(PUT 会 404 -- entry 不存在)。

## 已应用 + 实时验证 (2026-07-12, m5max)

通过 admin API 应用 (`auto-login?key=<主 api key>` 换 session cookie, 然后
`PUT /admin/api/models/qwen3-asr-1.7b-audio8-text4/settings` 带
`{"aligner_model": "qwen3-forced-aligner-0.6b-4bit"}`)。应用时 `active_requests`
为 0。写入 live 生效, 无 reload, 无重启。

注意 aligner 的值必须用 entry id `qwen3-forced-aligner-0.6b-4bit`, 不能用规范名
`Qwen3-ForcedAligner-0.6B-4bit` -- 后者解析不到 (大小写无关匹配在
`forcedaligner` vs `forced-aligner` 的连字符上失败), 所以旧的孤立条目是双重坏的
(外层 key 错 + aligner 值也无法解析)。

运行中 server 上的实测结果:

- 单声道 60s 片段, `word_timestamps=true`: 138 字, 114 个词级时间戳 (修前是 0)。
- 立体声 60s 片段, `energy_tripass` + `left_speaker=销售` /
  `right_speaker=客户`: 19 个说话人轮次, 销售/客户均衡, 词级时间戳 (修前完全为空)。
- 完整 480s 通话, `energy_tripass` + `on_aligner_overflow=chunk`: 108 轮,
  1721 词, 销售 54 / 客户 54 轮, 覆盖到 479.1s, 开场到收尾连贯对话。3-pass +
  分块对齐约 76s 墙钟。

## 相关风险: settings key drift (仅记录, 本次不修)

这是一个系统性模式的一个实例: m5max 的 `model_settings.json` 里 27 个 settings
key 有 20 个不匹配任何当前 entry id。模型随时间被改名 / 重注册, 旧 settings 条目
被遗留在旧名字下。任何这样的模型都会静默丢掉**全部** per-model 配置 (context
window, thinking, dflash, aligner, ...), 因为每个 `resolve_model_id` ->
`get_settings` 消费者都按 entry id 查。

不要用 `get_settings` 里的模糊匹配来修 -- 这些漂移的 key 是近似重名
(`gemma4-e2b` / `gemma4-e2b-4bit`, `qwen-dense-9b` / `qwen-dense-4b`), 任何
去后缀 / 大小写无关的兜底都可能把 stale 配置错配到别的模型。安全的清理是一次独立
的显式审计: 对每个孤立 key, 要么 re-key 到匹配的 entry id, 要么删除, 逐条处理,
人工把关。

审计结论 (2026-07-12, m5max): 20 个孤立 key 全部携带真实非默认配置 (dflash /
specprefill draft 路径, reasoning parser, thinking budget, 采样, mtp,
model_type_override) -- 没有一个是全默认死配置。只有一个有明确的 1:1 当前 entry
对应 (`Qwen3-ASR-...-mlx` -> `qwen3-asr-1.7b-audio8-text4`), 就是上面的 aligner
修复。其余 19 个没有精确的当前 entry 匹配; 模糊候选有歧义 (如 `gemma4-moe-26b-a4b`
可能对应 q4 / q6 / q8 三种量化 -- 把旧 dflash draft 应用到错的量化可能崩溃或降质),
需要只有 owner 才有的改名历史。部分 key 引用的 dflash draft 目录还在盘上
(gemma4-dense-31b-dflash-draft, gemma4-moe-26b-a4b-dflash-draft), 有些已消失
(gemma4-26b-a4b-draft, gemma4-moe-26b-a4b-assistant)。建议流程: owner 逐条确认每个
旧 key -> 当前 entry 的映射 (或"删除"), 然后走 admin PUT settings 路径 re-key, 让写入
落在 entry id 上。未自动应用。

## Cosmetic (无关)

`duration` 字段返回一个很小的残值 (~0.3s) 而不是音频时长, 在正常转写上也如此。与
aligner 无关; 本次不处理。
