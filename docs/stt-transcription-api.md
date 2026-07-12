# STT Transcription API — Integration Guide

Client integration reference for the speech-to-text endpoint on flyto MLX.
Covers auth, health, parameters, response shape, and long-audio handling.

English version first. Chinese translation follows below.

For speaker-attribution (diarization) depth, see `stt-diarize-client-api.md`.

---

## English

### 1. Endpoint and auth

```
POST  <base_url>/v1/audio/transcriptions
```

| Environment | `<base_url>` |
|---|---|
| Production (m5max) | `http://M5Max.local:8000` |

The request body is `multipart/form-data`. Auth is by API key, either header
(Bearer is the OpenAI-standard form and works with the OpenAI SDK):

```
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

Missing/invalid key returns 401. The server is OpenAI-compatible, so an OpenAI
SDK pointed at `base_url = http://M5Max.local:8000/v1` works directly.

### 2. Health check

```
GET  <base_url>/health          # no auth
```

Returns 200 with:

```json
{ "status": "healthy", "default_model": "...", "engine_pool": { "model_count": 33, "loaded_count": 1 }, "mcp": null }
```

Use it for liveness / readiness probes — no API key required. `status` is
`"healthy"` when the server is up.

### 3. Capabilities

- **Any-length transcription.** Long calls (tens of minutes to over an hour)
  transcribe in a single request without hitting the model's long-audio
  degeneration — handled automatically, no parameter needed.
- **Speaker attribution.** Stereo calls (one speaker per L/R channel, e.g.
  FreeSWITCH 2-leg recordings) return a sales/customer turn-by-turn transcript.
- **Word-level timestamps.** Optional per-word start/end times.
- **Subtitles.** SRT / VTT output.
- **Domain biasing.** A `prompt` steers proper nouns / product names.

### 4. Request parameters

**Core**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | Audio (wav/mp3/m4a/mp4/…). WAV recommended for long calls (see notes). |
| `model` | string | required | `qwen3-asr-1.7b-audio8-text4` |
| `language` | string | auto | Language code, e.g. `zh`. Omit to auto-detect. |
| `prompt` | string | — | Transcription-biasing context (domain vocabulary, proper nouns). |
| `response_format` | string | `json` | `json` / `verbose_json` / `text` / `srt` / `vtt` |

**Long audio (on by default — usually left alone)**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `long_audio` | string | `auto` | `auto` = transcribe once, auto-re-chunk only if it degenerates; `chunk` = always chunk; `off` = single pass, never chunk. |
| `chunk_minutes` | float | `15` | Target minutes per window when chunking. |

**Word timestamps / diarization**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `word_timestamps` | bool | `false` | Return per-word start/end. |
| `diarize_backend` | string | `auto` | `energy_tripass` = stereo L/R → speakers (best for 2-leg calls); `pyannote` = mono multi-speaker; `none`. |
| `left_speaker` / `right_speaker` | string | — | Speaker labels for the L / R channel under `energy_tripass` (both required). |

Advanced sampling params (`temperature`, `top_p`, `repetition_penalty`,
`max_tokens`, …) are also accepted but rarely needed. The full, machine-readable
field list with descriptions is at `<base_url>/openapi.json` and the browsable
Swagger UI at `<base_url>/docs`.

### 5. Response

`json` (default) / `verbose_json`:

```json
{
  "text": "full transcript",
  "language": "zh",
  "duration": 2504.0,
  "segments": [
    {
      "start": 0.0,
      "end": 12.3,
      "text": "segment text",
      "speaker": "sales",
      "words": [ { "word": "hi", "start": 0.2, "end": 0.4 } ]
    }
  ]
}
```

`speaker` appears only with diarization; `words` only with `word_timestamps`.
`response_format=text` returns the transcript body as plain text with no
envelope. `srt` / `vtt` return subtitle text.

### 6. Examples

Basic — any length, auto-handled:

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  -F file=@call.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  http://M5Max.local:8000/v1/audio/transcriptions
```

Stereo sales call — sales/customer turns + word timestamps:

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  -F file=@call-stereo.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  -F diarize_backend=energy_tripass \
  -F left_speaker=sales -F right_speaker=customer \
  -F word_timestamps=true \
  -F response_format=verbose_json \
  http://M5Max.local:8000/v1/audio/transcriptions
```

### 7. Notes and limits

- **Upload limit: 300 MB.** Stereo 8 kHz WAV is ~1.9 MB/min → ~160 min stereo
  fits; mono is ~half the size (~320 min). Beyond that, downmix or split.
- **Mono vs stereo.** For a plain transcript, upload mono (smaller — the server
  down-mixes for ASR anyway). Stereo is required only for `energy_tripass`
  speaker separation.
- **WAV for long calls.** Any format transcribes, but the long-audio auto-
  recovery re-chunk step decodes via libsndfile (WAV). A very long non-WAV file
  that degenerates may not auto-recover. FreeSWITCH recordings are WAV.
- **One transcription per request** (`n` must be 1).

---

## 中文

说话人分离的完整细节见 `stt-diarize-client-api.md`。

### 1. 端点与鉴权

```
POST  <base_url>/v1/audio/transcriptions
```

| 环境 | `<base_url>` |
|---|---|
| 生产 (m5max) | `http://M5Max.local:8000` |

请求体是 `multipart/form-data`。鉴权用 API key，两种头任选（Bearer 是 OpenAI
标准形式，OpenAI SDK 可直接用）：

```
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

缺失/错误 key 返回 401。服务器 OpenAI 兼容，OpenAI SDK 指向
`base_url = http://M5Max.local:8000/v1` 即可直接用。

### 2. 健康检测

```
GET  <base_url>/health          # 无需鉴权
```

返回 200：

```json
{ "status": "healthy", "default_model": "...", "engine_pool": { "model_count": 33, "loaded_count": 1 }, "mcp": null }
```

用于探活 / 就绪探测 — 不需要 API key。服务器正常时 `status` 为 `"healthy"`。

### 3. 能力

- **任意长度转写。** 长通话（几十分钟到一个多小时）一条请求转完，不会撞模型
  长音频退化 — 自动处理，无需参数。
- **说话人分离。** 立体声通话（左右声道各一人，如 FreeSWITCH 双腿录音）返回
  销售/客户分轮对话。
- **词级时间戳。** 可选，每个词的起止时间。
- **字幕。** SRT / VTT 输出。
- **领域偏置。** `prompt` 引导专名 / 产品名。

### 4. 请求参数

**核心**

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `file` | 文件 | 必填 | 音频 (wav/mp3/m4a/mp4/…)。长通话建议 WAV (见注意事项)。 |
| `model` | string | 必填 | `qwen3-asr-1.7b-audio8-text4` |
| `language` | string | 自动 | 语言码，如 `zh`。不传则自动检测。 |
| `prompt` | string | — | 转写偏置上下文 (领域词、专名)。 |
| `response_format` | string | `json` | `json` / `verbose_json` / `text` / `srt` / `vtt` |

**长音频 (默认已开，一般不动)**

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `long_audio` | string | `auto` | `auto` = 转一趟、崩了才自动切块救回；`chunk` = 总是切块；`off` = 单趟从不切。 |
| `chunk_minutes` | float | `15` | 切块时每块目标分钟数。 |

**词时间戳 / 说话人分离**

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `word_timestamps` | bool | `false` | 返回每词起止时间。 |
| `diarize_backend` | string | `auto` | `energy_tripass` = 立体声左右声道分说话人 (双腿通话首选)；`pyannote` = 单声道多说话人；`none`。 |
| `left_speaker` / `right_speaker` | string | — | `energy_tripass` 下左/右声道的说话人名 (两个都要传)。 |

进阶采样参数 (`temperature`、`top_p`、`repetition_penalty`、`max_tokens` 等)
也支持，一般不用。完整机器可读字段列表带说明在 `<base_url>/openapi.json`，可视化
Swagger 在 `<base_url>/docs`。

### 5. 响应

`json` (默认) / `verbose_json`：

```json
{
  "text": "完整转写文本",
  "language": "zh",
  "duration": 2504.0,
  "segments": [
    {
      "start": 0.0,
      "end": 12.3,
      "text": "这一段的文本",
      "speaker": "销售",
      "words": [ { "word": "喂", "start": 0.2, "end": 0.4 } ]
    }
  ]
}
```

`speaker` 仅分离时有；`words` 仅 `word_timestamps` 时有。`response_format=text`
返回纯文本 body 无外层结构。`srt` / `vtt` 返回字幕文本。

### 6. 示例

基础 — 任意长度自动兜底：

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  -F file=@call.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  http://M5Max.local:8000/v1/audio/transcriptions
```

立体声电销 — 销售/客户分轮 + 词级时间戳：

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  -F file=@call-stereo.wav \
  -F model=qwen3-asr-1.7b-audio8-text4 \
  -F diarize_backend=energy_tripass \
  -F left_speaker=销售 -F right_speaker=客户 \
  -F word_timestamps=true \
  -F response_format=verbose_json \
  http://M5Max.local:8000/v1/audio/transcriptions
```

### 7. 注意事项与上限

- **上传上限 300 MB。** 立体声 8 kHz WAV 约 1.9 MB/分钟 → 约 160 分钟立体声可
  容纳；单声道约一半 (约 320 分钟)。超出请降混或分段。
- **单声道 vs 立体声。** 纯转写上传单声道即可 (更小 — 服务器 ASR 本就降混)。
  只有 `energy_tripass` 分说话人才需要立体声。
- **长通话用 WAV。** 任何格式都能转，但长音频自动救回的切块步骤走 libsndfile
  (WAV) 解码；超长非 WAV 文件万一退化可能不自动救回。FreeSWITCH 录音是 WAV。
- **每请求一路转写** (`n` 必须为 1)。
