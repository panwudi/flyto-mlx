# fmlx Qwen inpaint engine spec (mask pixel-lock editing)

状态: Phase 1 进行中 (feat/image-inpaint). 本 spec 是实现前的设计基线, 算法事实
对照 mflux 0.18.14 (mlx-gen) 源码 + diffusers `QwenImageInpaintPipeline`
(PR huggingface/diffusers#12117) 核实. 代码尚未真机验证, 出口判据见 section 8.

## 0. 背景与定位

外部创作引擎以 fmlx 作本地图像后端, 硬需求是产品像素级锁定: 广告产品图主体
(logo / 文字 / 形状) 必须逐像素不变, 只换背景. 现有 qwen-image-edit-2511 是语义
软锁 (无 mask), 满足不了广告红线. 引入 mask 驱动的 inpaint 来做硬锁.

两段式落地:

- Phase 1 (本 spec): 移植 diffusers `QwenImageInpaintPipeline` 到 mflux, 跑在
  已有的 qwen-image 基座权重上 (Apache-2.0, m5max 已注册 qwen-image-2512-4bit).
  打通 fmlx 这侧全部共用管道 (mask 字段 / 路由 / worker 分支 / 像素合成 / 租约).
- Phase 2: 在 Phase 1 的管道上加 InstantX/Qwen-Image-ControlNet-Inpainting
  (Apache-2.0) 的 ControlNet 分支换质量. ControlNet 是独立注册模型 (基座 + control
  权重), 复用 Phase 1 的 API / mask / 合成, 只新增一个 pipeline 分支.

定位: fmlx 自有功能, 不回流上游 (soft-fork). qwen-edit 保留, inpaint 并存,
按场景路由由调用方负责 (产品锁 -> inpaint 硬锁; 角色锁 / 换景 -> qwen-edit 软锁).

## 1. 算法 (diffusers inpaint 移植到 mflux flow-match)

基座 inpaint 不需要专门的 inpaint checkpoint, 它是 img2img + 逐步 mask blend.
mflux 的 `QwenImage.generate_image` 已经做了 img2img (VAE 编码原图 -> 按
image_strength 加噪成初始 latent -> 从 init_time_step 起跑), 所以移植量集中在
mask blend 这一步.

diffusers 参考实现 (已核实):

```
# 起始步 (strength -> 截断时间表)
init_timestep = min(num_steps * strength, num_steps)
t_start       = max(num_steps - init_timestep, 0)
timesteps     = scheduler.timesteps[t_start:]

# 初始 latent
image_latents = vae_encode(image)                 # 原图干净 latent
noise         = randn(seed)
latents       = scale_noise(image_latents, t0, noise)

# mask: 下采样到 latent 分辨率, white=1=重画, black=0=保留
mask = interpolate(mask, latent_h, latent_w)

# 去噪循环, 每步在 scheduler.step 之后:
init_latents_proper = image_latents
if i < len(timesteps) - 1:
    init_latents_proper = scale_noise(image_latents, t[i+1], noise)   # 末步用干净 latent
latents = (1 - mask) * init_latents_proper + mask * latents
```

flow-match 下 `scale_noise(x0, noise, t) = (1 - sigma_t) * x0 + sigma_t * noise`,
mflux 的 `config.scheduler.sigmas` 直接给逐步 sigma. 所以 mflux 移植:

- `image_latents`: 用 `QwenLatentCreator` + VAE 编码原图得到 packed latent
  (复用 `Img2Img` 的编码路径, 但要单独留住干净 image_latents 与 noise, 因为每步
  都要用它们重算 init_latents_proper).
- `mask`: 加载 mask 图 -> 下采样到 latent 分辨率 -> 按 `QwenLatentCreator` 的
  packed 序列布局打包 (`[B, seq, C]`), 在 channel 维广播.
- 每步 blend: `latents = (1 - m) * ((1 - sig_next) * image_latents + sig_next * noise) + m * latents`,
  末步 `sig_next = 0`.
- 其余 (prompt 编码 / transformer 前向 / guided noise / scheduler.step) 与 mflux
  `QwenImage` 循环逐字一致, 不动.

## 2. Mask 极性 (关键, 否则功能静默反转)

diffusers 约定: mask white=1=重画, black=0=保留.

调用方用 BiRefNet 出 mask, BiRefNet 输出的是前景 (产品) mask: 产品=白=1. 若直接
喂给 diffusers 语义, 会重画产品 / 保留背景, 正好相反, 且看起来像质量问题不像 bug.

fmlx 契约 (写死, 文档显式声明) -- 采用 diffusers / inpaint 业界标准约定:

- inpaint 端点接收的是标准 inpaint mask: 白色 (255) = 要重画的区域 (背景),
  黑色 (0) = 要保留锁住的区域 (产品).
- 反转责任在调用方: BiRefNet 出的是前景 mask (产品=白). 调用方在 C# 端 BiRefNet
  之后取反 (产品->黑保留, 背景->白重画), 再发给 fmlx. fmlx 侧不做任何反转, 收到的
  mask 直接喂给 section 1 算法 (那里的 mask 已是 white=1=重画).
- 理由: 与 diffusers / SD / ComfyUI 生态一致, fmlx 侧零反转零歧义, 任何未来客户端
  都按业界标准理解; 调用方反转成本极低且已确认承担 (2026-06-17 联调对齐).

这条必须进 API 契约文档 (image-video-generation-client-api.md inpaint 段), 并在
端点收到全白 / 全黑 mask 时给明确提示 (易错信号).

## 3. 像素级锁 = latent blend + 末尾像素合成

latent blend 让保留区全程贴着原图轨迹走, 但整图过 VAE 编解码 (有损), 保留区不是
字节级不变. 广告红线要字节级, 所以 worker 在 VAE 解码出图后, 在像素空间再合成一次
(mask 为标准约定 m, 白=1=重画, 黑=0=保留):

```
out = (1 - m) * original_pixels + m * generated_pixels
```

即保留区 (m=0) 取原图像素, 重画区 (m=1) 取模型输出. 边界做 1-2px 羽化避免硬切观感.
合成后产品区 = 原图像素, 严格不变; 背景区 = 模型重画.

## 4. Dispatch 设计 (Phase-2 形状, Phase 1 就定型)

不要用 "有 mask 就 inpaint" 这种推断式路由, 否则 Phase 2 的 ControlNet (独立模型)
要重写路由. 路由从 model entry 的 `image_pipeline` 解析:

- `image_pipeline` 取值扩展为 `{t2i, edit, inpaint, controlnet_inpaint}`.
- 路由: entry.image_pipeline 属于 inpaint 类 (`inpaint` / `controlnet_inpaint`)
  时, 要求恰好 1 张 image + 1 张 mask, 走 inpaint 提交路径.
- worker 按 `normalized["pipeline"]` 分支: `inpaint` -> 基座 Qwen inpaint loop
  (Phase 1); `controlnet_inpaint` -> ControlNet inpaint loop (Phase 2), 多吃一个
  controlnet 权重路径.
- ControlNet 作为独立注册模型 (基座 + control 权重) 插入, 只加一个 pipeline 分支
  与一个权重路径, 不动 dispatch. 这是 Phase 1 要打的地基.

### 4.1 模型注册 (Phase 1)

基座 inpaint 跑在已有的 qwen-image 基座权重上, Phase 1 不需要新下权重. 需要一个
`image_pipeline=inpaint` 的逻辑模型条目指向 qwen-image 基座权重目录. 注册机制
(settings 标记 / registry 别名映射 logical-id -> (weights_dir, pipeline)) 在实现期
定稿, 但 dispatch 契约 (路由读 entry.image_pipeline, worker 读 normalized.pipeline)
现在锁死.

## 5. normalized params 契约 (route -> worker)

inpaint 任务的 normalized dict 在现有 image 字段基础上新增:

- `pipeline`: `"inpaint"` | `"controlnet_inpaint"`
- `image_paths`: `[原图路径]` (恰好 1 张)
- `mask_path`: 标准 inpaint mask 路径 (白=重画 / 黑=保留, fmlx 不反转, 调用方已反转)
- `image_strength`: inpaint 重画强度, 默认 1.0 (产品锁场景背景完全重画; 保留靠
  mask 而非 strength)
- Phase 2 追加: `controlnet_path`, `controlnet_conditioning_scale`

mask 与 image 一样走模型外提取 (multipart 文件字段 / JSON base64), 不进
pydantic 模型. route 提取后落临时文件, 路径进 normalized.

## 5.1 客户端用法 (curl)

复用 `POST /v1/images` (不开新端点), model 指向注册的 inpaint 模型, 带 image +
mask. mask 是标准约定 (白=重画/黑=保留), 调用方已反转 BiRefNet 前景.

```bash
# 产品锁: 锁住产品 (mask 里产品=黑), 只重画背景 (背景=白)
curl -s http://HOST:8000/v1/images \
  -H "Authorization: Bearer $FMLX_KEY" \
  -F model=qwen-image-inpaint-4bit \
  -F image=@product.png \
  -F mask=@mask_inverted.png \
  -F 'prompt=the product on a bright marble kitchen counter, soft daylight' \
  -F sync=true
# 返回 OpenAI 形状 data[].b64_json; 产品区与原图逐像素一致, 背景按 prompt 重画
```

JSON body 同 image: `"image"` / `"mask"` 为 data URL 或 base64 字符串.
prompt 用整图描述式 (ControlNet 阶段尤其), 不是指令式. 完整 client-API 文档的
inpaint 段在 PR #84 (image-video-generation-client-api.md) 合并后补.

## 6. 文件改动清单 (Phase 1)

- `omlx/api/image_models.py`: 无需加字段 (mask 走模型外提取); 可加注释说明.
- `omlx/api/image_routes.py`: 提取 `mask` 字段 (multipart 文件 / JSON base64,
  同 `image` 的解码路径); inpaint 类 pipeline 的校验 (恰好 1 image + 1 mask);
  normalized 写入 mask_path / pipeline.
- `omlx/image/worker.py`: 加 inpaint 分支, 调用新模块.
- `omlx/image/qwen_inpaint.py` (新): 组合 mflux 组件的 masked-denoise loop +
  像素合成. 不 monkeypatch 安装好的 mflux, 在 fmlx worker 侧组合其组件.
- `docs/image-video-generation-client-api.md`: 补 inpaint 段 (mask 极性契约).

## 7. 内存租约

沿用现有 image 租约 (qwen-image 基座 ~26GB, image_routes `_DEFAULT_LEASE_GB`).
inpaint 多一次 VAE 编码原图 + mask, 峰值与 edit 同量级, Phase 1 复用 qwen-image
的 lease, 实现期真机标定确认 (video spec 9.1 纪律).

## 8. 验证出口判据 (Phase 1 完成定义)

单测过 != 真机过. 核心 loop 的完成判据是 m5max 真机验证.

### 8.1 验证方法 (为何不是 diffusers 对拍)

原计划对拍真 diffusers `QwenImageInpaintPipeline`, 但 m5max 全机无 diffusers,
且现存权重是 mflux 4bit 量化版 (非 HF torch 原版); 真对拍要装 diffusers + 下
~40GB torch 权重, 不划算且属大动作. 改用 mflux 自带 txt2img 作 ground truth 的
自洽验证, 覆盖同样的失败模式 (保留区漂移 / sigma 索引 / mask 打包):

- 全白 mask (整图重画) 应 == mflux QwenImage txt2img (同 seed/prompt).
- 全黑 mask (整图保留) 应 == 原图 (仅 VAE 往返).
- 小块 mask 保留区应贴近原图, 重画区随 prompt 变.

### 8.2 验证结果 (2026-06-17, m5max, qwen-image-2512-4bit, 512px/8步)

PASS. 实测 (MSE, 0-255):

- allblack vs original = 1.4 (保留全部 ~= identity; 证 blend/mask 打包/sigma 索引对)
- allwhite vs txt2img  = 0.0 (重画全部与 mflux txt2img 逐字节相同; 证去噪路径对)
- small vs original    = 9072.9, baseline txt2img vs original = 15198.2 (保留区拉近原图)
- 视觉: 红方块产品位置/边缘/颜色完整保留, 背景重画成热带海滩.

生产模块 (含像素合成 + progress 回调 + 可选尺寸) 复跑 PASS:

- 产品内部 max|diff| = 0 (字节级锁定, 合成后产品像素与原图逐像素一致).
- 背景 mean|diff| = 108 (完全重画).
- progress 回调 [1..8] 正常 (worker stall 超时安全).
- 模块按文件路径加载, 仅依赖 mflux/mlx/numpy/PIL, 不引 omlx (worker venv HARD RULE).

panic 纪律: 验证脚本绕过 MediaJobManager 单租约, 会重演单请求撞顶拖死整机的风险
(m5max-ops-panic-reality). 实测时机 = server idle (95% free, 无大模型驻留), 脚本
自身 set_wired_limit(30GB) + cache 1GB; 单次 512px/8步约 10s.

### 8.3 剩余 (端到端可用前)

1. 注册一个 image_pipeline=inpaint 的逻辑模型指向 qwen-image 基座权重 (discovery
   或 registry), 路由才能 dispatch.
2. 走 MediaJobManager 单租约的端到端任务跑通 (queued -> in_progress -> completed
   -> content), 在 m5max 真机 (部署分支 + 重启 server) A/B.
3. 客户端 API 文档补 inpaint 段 (mask 极性契约).
