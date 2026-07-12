# STT Long-Audio Auto-Chunking

How the `/v1/audio/transcriptions` endpoint transcribes calls of any length
without hitting the decoder degeneration that wrecks a single long pass.

English version first. Chinese translation follows below.

---

## English

### 1. The problem

Decoder-only ASR models (Qwen3-ASR and family) lose the thread on a long
single pass and collapse into a repeated-token loop -- a Chinese call
degenerates into `啊。啊。啊。...` -- dropping the rest of the audio. This is
not a token-budget problem: raising `max_tokens` or `repetition_penalty` only
reshapes the garbage (the loop runs longer, or switches to a longer repeated
phrase). The model has lost the thread, not run out of room.

The trigger is **output length, not audio minutes** -- the model degenerates
after generating roughly a fixed amount of text, so a dense, fast-talking call
hits it sooner in wall-clock than a slow one. Measured on
`qwen3-asr-1.7b-audio8-text4` (mono down-mix, single pass):

- A slow 20-min call: 5593 chars, uniqueness 1.00, clean.
- A dense 42-min call, whole file: 8446 chars, uniqueness 0.09 -- front half
  correct, back half `啊。啊。` loop.
- The same dense call's first 7 min alone: 1932 chars, clean; its 7-14 min
  slice alone: 2216 chars, clean; but its first 14 min together: 8436 chars,
  degenerate. Each slice is fine on its own -- only the accumulated output of
  the longer window trips the loop.

Because the threshold is content density, no single window length is safe for
every call. Chunking bounds the per-window output, and the repeat-loop guard
(below) adapts: it detects a window that still degenerated and re-splits it,
so a dense call converges to short-enough windows automatically.

### 2. The fix

Split long audio at natural silence pauses into windows the model handles
cleanly, transcribe each window **independently**, and concatenate with
per-window time offsets. Three details matter:

1. **Cut at silence, not at a fixed clock.** A fixed-seconds cut lands
   mid-word ~2.3 words out of place on average; a silence-anchored cut lands
   within a fraction of a word. Silence detection is a small energy analyzer
   (frame RMS + a relative threshold) -- no extra model, no dependency beyond
   numpy.
2. **No cross-window context.** Each window is transcribed on its own with no
   `previous_text`. Conditioning on previous text is exactly what seeds the
   Whisper-family repeat loop; feeding the model the tail of a degenerate
   window would propagate the loop forward.
3. **A repeat-loop guard.** After each window is transcribed its 12-gram
   uniqueness is checked. A window that still degenerated (uniqueness below
   0.5) is re-split at its midpoint silence and re-transcribed. This bottoms
   out at a 5-minute floor and a recursion cap, so genuinely repetitive but
   valid audio never triggers endless splitting -- worst case you get the
   best-effort transcript, never a hang or a 500.

Windows are cut at silence, so there is no need for inter-window overlap
(overlap raises word error rate); the concatenation is a straight ordered
join.

**Verified end-to-end** on the dense 42-min call. The planner produced three
14-min windows; each one degenerated on its first pass (uniqueness
0.09 / 0.12 / 0.31), the guard re-split each at its mid-silence, and all six
7-min leaves came back clean (uniqueness 1.00). Merged result: 12604 chars,
uniqueness 1.00, a proper call opening and closing -- versus the 8446-char,
uniqueness-0.09 degenerate whole-file pass.

Note that the chunk count is `ceil(total / chunk_minutes)`, so 42 min splits
into 3 windows at both the 15-min and 20-min defaults (42/15 and 42/20 both
round up to 3) -- the same ~14-min windows, and the same guard re-splits on a
dense call. The default only changes the window count where the rounding
differs (e.g. a 50-min call is 3 windows at 20 but 4 at 15). Because this call
is dense, every window tripped the guard: one wasted degenerate pass per
window before the re-split. Correctness does not depend on the default -- the
guard converges regardless -- but a smaller default reduces those wasted
passes on dense calls (measured clean at <=7-min windows), at the cost of more
windows on every call. It makes no difference to slow calls, whose windows
pass on the first try.

### 3. It just works -- the server decides

The default is `long_audio=auto`: the server transcribes once, checks the
output for the repeat-loop signature (the same 12-gram uniqueness check the
guard uses), and **only if it degenerated** re-transcribes chunked. A bare
curl transcribes a call of any length with nothing extra:

```bash
curl -H "Authorization: Bearer $KEY" \
  -F file=@call-89min.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  http://m5max:8000/v1/audio/transcriptions
```

`auto` is zero-config and model-agnostic: clean, short, or non-degenerating
audio (e.g. Whisper) never triggers a re-chunk, so it costs one extra pass
only when a call actually broke. The consumer never has to know the feature
exists.

### 4. Overriding per request

The `long_audio` and `chunk_minutes` form fields let a consumer override the
default. They are published in the server's OpenAPI schema (`/openapi.json`,
Swagger `/docs`).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `long_audio` | string | `auto` | `auto` = transcribe once, re-chunk only if degenerate; `chunk` = always chunk proactively (skips the probe -- best on a model known to degenerate); `off` = single pass, never chunk. |
| `chunk_minutes` | float | `15` | Target window length in minutes for chunking. The window count is `ceil(total / chunk_minutes)`. Windows snap to the nearest silence near each boundary; a hard cap of 1.25x (18.75 min at the default) bounds any single window. |

### 5. Per-model default

`auto` is the server default for every model. To pin a specific mode on a
model -- e.g. set a known-degenerating model to `chunk` so it skips the
single-pass probe entirely (no wasted pass) -- use the admin settings API:

```
PUT <base_url>/api/models/{model_id}/settings
{ "default_long_audio": "chunk", "long_audio_chunk_minutes": 15 }
```

`default_long_audio` is `auto`, `chunk`, or `off`; a per-request `long_audio`
field always wins over it. `long_audio_chunk_minutes` sets the target window
length when omitted per request.

### 6. Composition with diarization and word timestamps

- **`energy_tripass` is not chunked.** The 3-pass stereo path already re-ASRs
  in short forced-aligner windows and rebuilds its text from those windows, so
  its merged transcript does not degenerate on long audio. Enabling
  `long_audio=chunk` on a tripass request is a no-op for the transcription
  split (tripass keeps its own path).
- **`word_timestamps` composes.** When word timestamps are requested with
  chunking, each transcript window runs the configured forced aligner on its
  own audio. A ~15 minute window still exceeds the aligner's ~270 s
  single-segment limit, so within a window the aligner sub-windows itself
  (the existing `on_aligner_overflow=chunk` machinery) -- no separate
  `on_aligner_overflow` field is needed, chunking implies it.
- **`energy` (1-pass) and `pyannote` compose.** Both run after the primary
  transcript is assembled and operate on global-timeline words, so they see
  the clean concatenated result exactly as they would a single pass.

### 7. Where it lives

- `omlx/engine/audio_chunk.py` -- pure, engine-free: `plan_chunks` (silence
  split planning), `ngram_uniqueness` (the guard), `bisect_at_silence`
  (re-split point), `offset_result_times` / `concat_chunk_results` (merge).
- `omlx/api/audio_routes.py` -- the `_transcribe_long_audio` window loop in
  `create_transcription`, plus the `long_audio` / `chunk_minutes` fields and
  their per-model settings resolution.

---

## 中文

### 1. 问题

decoder-only 的 ASR 模型 (Qwen3-ASR 一族) 长音频单趟转写会丢线索, 退化成重复
token 循环 -- 中文通话退化成 `啊。啊。啊。...` -- 后面内容全丢. 这不是 token
额度问题: 加 `max_tokens` 或 `repetition_penalty` 只改垃圾的形状 (循环更久, 或
换成更长的重复短语). 是模型丢了线索, 不是没地方写.

诱因是**输出长度, 不是音频分钟数** -- 模型生成到大约固定字数就退化, 故语速快
密度高的通话在墙钟上更早撞到. 实测 `qwen3-asr-1.7b-audio8-text4` (单声道降混,
单趟):

- 一通慢速 20 分钟通话: 5593 字, 唯一率 1.00, 干净.
- 一通密集 42 分钟通话整文件: 8446 字, 唯一率 0.09 -- 前半正常后半 `啊。啊。`
  循环.
- 同一通密集通话前 7 分钟单独转: 1932 字, 干净; 7-14 分钟片段单独转: 2216 字,
  干净; 但前 14 分钟一起转: 8436 字, 退化. 每个片段单独都没问题 -- 只有更长窗口
  累积的输出才触发循环.

因为阈值取决于内容密度, 没有哪个固定窗口时长对所有通话都安全. 分块限制每窗输出,
重复守卫 (见下) 自适应: 检测到仍退化的窗口就再切, 密集通话自动收敛到足够短的
窗口.

### 2. 解法

在自然静音处把长音频切成模型能干净处理的窗口, 每个窗口**独立**转写, 按窗口
起点做时间戳偏移后顺序拼接. 三个关键点:

1. **在静音处切, 不按固定秒数切.** 固定秒数切平均每刀错约 2.3 个词; 静音锚定
   的切点误差在一个词以内. 静音检测是个小能量分析器 (帧 RMS + 相对阈值) --
   不加模型, 除 numpy 外无依赖.
2. **绝不传跨窗上下文.** 每个窗口独立转写, 不带 `previous_text`. 条件依赖前文
   正是 Whisper 一族重复循环的诱因; 把一个退化窗口的尾巴喂给模型会把循环往后
   传染.
3. **重复守卫.** 每个窗口转完检查 12-gram 唯一率. 仍退化 (唯一率低于 0.5) 的
   窗口在其中点静音处再对半切重转. 有 5 分钟下限和递归深度上限兜底, 所以真正
   重复但有效的音频不会触发无限切分 -- 最坏情况是拿到尽力而为的转写, 绝不卡死
   也绝不 500.

窗口在静音处切, 故无需窗间重叠 (重叠会拉高词错率); 拼接就是顺序直连.

**已端到端验证**: 用那通密集 42 分钟通话实测. 规划器切出三个 14 分钟窗口, 每个
首趟都退化 (唯一率 0.09 / 0.12 / 0.31), 守卫在各自中点静音处再切, 六个 7 分钟
叶子全部干净 (唯一率 1.00). 合并结果: 12604 字, 唯一率 1.00, 开头结尾都正常 --
对比退化整文件的 8446 字 / 唯一率 0.09.

注意窗口数 = `ceil(总时长 / chunk_minutes)`, 所以 42 分钟在 15 分钟和 20 分钟
默认下都切成 3 块 (42/15 和 42/20 都向上进位到 3) -- 同样的 ~14 分钟窗口, 密集
通话上同样触发守卫再切. 默认只在进位不同的时长上改变块数 (比如 50 分钟通话在
20 分钟下 3 块, 15 分钟下 4 块). 因为这通密集, 每个窗口都触发了守卫: 每窗多一趟
退化的白跑再切. 正确性不依赖默认 -- 守卫都会收敛 -- 但更小的默认能减少密集通话
上这些白跑 (实测 <=7 分钟窗口干净), 代价是每通话都切更多块. 对慢速通话无差别
(它们的窗口首趟就过).

### 3. 开箱即用 -- 服务器自己判断

默认是 `long_audio=auto`: 服务器先正常转一趟, 用重复率检查 (守卫用的同一个
12-gram 唯一率) 看输出像不像崩溃循环, **只有崩了才**切块重转. 裸 curl 什么都
不用加就能转任意长度:

```bash
curl -H "Authorization: Bearer $KEY" \
  -F file=@call-89min.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  http://m5max:8000/v1/audio/transcriptions
```

`auto` 零配置、模型无关: 干净/短/不退化的音频 (比如 Whisper) 永远不会触发重转,
所以只有真崩的通话才多花一趟. 消费者根本不需要知道这个特性存在.

### 4. 按请求覆盖

`long_audio` 和 `chunk_minutes` 两个表单字段让消费者覆盖默认. 它们发布在服务器
的 OpenAPI 规范里 (`/openapi.json`, Swagger `/docs`).

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `long_audio` | string | `auto` | `auto` = 转一趟, 崩了才重转切块; `chunk` = 总是主动切块 (跳过试探趟, 适合已知会崩的模型); `off` = 单趟, 从不切块. |
| `chunk_minutes` | float | `15` | 切块的目标窗口时长 (分钟). 窗口数 = `ceil(总时长 / chunk_minutes)`. 窗口边界吸附到附近静音; 硬上限为 1.25 倍 (默认下 18.75 分钟) 约束单窗口. |

### 5. 按模型设默认

`auto` 是所有模型的服务器默认. 想给某个模型钉死某个模式 -- 比如把已知会崩的
模型设成 `chunk` 让它跳过单趟试探 (不白跑) -- 通过 admin 设置 API:

```
PUT <base_url>/api/models/{model_id}/settings
{ "default_long_audio": "chunk", "long_audio_chunk_minutes": 15 }
```

`default_long_audio` 取 `auto`, `chunk`, 或 `off`; 按请求的 `long_audio` 字段
永远优先. `long_audio_chunk_minutes` 在请求未指定时作为目标窗口时长.

### 6. 与 diarization / 词级时间戳的组合

- **`energy_tripass` 不分块.** 3-pass 立体声路径本就在短的 forced-aligner
  窗口里重跑 ASR 并用这些窗口重建文本, 故其合并转写在长音频上不退化. 在
  tripass 请求上开 `long_audio=chunk` 对转写分块是 no-op (tripass 走自己的
  路径).
- **`word_timestamps` 可组合.** 分块时请求词级时间戳, 每个转写窗口在自己的
  音频上跑配置的 forced aligner. 一个约 15 分钟窗口仍超过 aligner 约 270s 的
  单段上限, 故窗口内 aligner 自行再分窗 (复用已有 `on_aligner_overflow=chunk`
  机制) -- 无需单独传 `on_aligner_overflow`, 分块隐含它.
- **`energy` (单 pass) 和 `pyannote` 可组合.** 两者都在主转写组装完之后运行,
  作用于全局时间轴的词, 故看到的是干净的拼接结果, 与单趟无异.

### 7. 代码位置

- `omlx/engine/audio_chunk.py` -- 纯逻辑, 不依赖引擎: `plan_chunks` (静音切点
  规划), `ngram_uniqueness` (守卫), `bisect_at_silence` (再切点),
  `offset_result_times` / `concat_chunk_results` (合并).
- `omlx/api/audio_routes.py` -- `create_transcription` 里的
  `_transcribe_long_audio` 窗口循环, 以及 `long_audio` / `chunk_minutes`
  字段和它们的按模型设置解析.
