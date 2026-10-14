# Flyto MLX menubar (Rust)

`Flyto MLX.app` is a status-bar menu that starts, adopts and supervises the local
`omlx.cli serve` process. It replaces the Swift shell in `apps/omlx-mac`, keeping
its bundle id (`com.panwudi.flyto-mlx`), install path, base-path resolution and
log location. The Swift shell stays in the tree until this one has run on m5max
for a while.

What it does differently from the Swift shell:

- If the port is already served by a healthy fmlx (for example after the app was
  killed and relaunched), it adopts that server instead of reporting a port
  conflict.
- Liveness is checked by PID, not by port. Stop sends SIGTERM and waits up to
  120 s for the process to be gone (it unloads models and flushes the SSD cache
  first) before SIGKILL. A new server is never spawned while an older one for
  the same base path is still exiting, so the new one does not count the old
  one's memory as external and pin its ceiling low.
- If the server PID disappears without a stop from the menu, it restarts after
  5, 10 and 20 s, then gives up until the server has been up for a minute.
- Stop, restart and quit ask for confirmation while requests are in flight.
- The menu shows loaded models, running and waiting requests and model memory.
- Menu text is Chinese or English, following the first macOS language.

Files: `src/supervisor.rs` (lifecycle), `src/procs.rs` (PID and port helpers),
`src/api.rs` (`/health`, `/admin/api/stats`), `src/main.rs` (menu),
`assets/menubar.svg` (template glyph), `assets/app-icon.svg` (app icon, same as
the admin UI favicon). Event log: `~/Library/Application Support/oMLX/logs/menubar.log`.
Server log: `.../oMLX/logs/server.log`.

Build and test:

```
cd apps/fmlx-menubar
cargo test --release
scripts/build.sh                      # -> build/Flyto MLX.app
OMLX_BASE_PATH=/tmp/fmlx-test OMLX_PORT=8796 target/release/fmlx-menubar   # dev run, no login item
```

Install on m5max (announce the restart first; the server is down for about a
minute):

```
# 1. Retire the old watchdog: it SIGKILLs the app whenever /health fails,
#    including during a deliberate restart, and orphans the server.
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.flyto.mlx.watchdog.plist
mv ~/Library/LaunchAgents/com.flyto.mlx.watchdog.plist ~/Library/LaunchAgents/com.flyto.mlx.watchdog.plist.disabled

# 2. Swap the bundle, keeping the old one for rollback. Quitting the old app
#    may leave its server running (the Swift shell only stops its own child);
#    that is fine, the new app adopts it.
osascript -e 'quit app "Flyto MLX"'
mv "/Applications/Flyto MLX.app" "/Applications/Flyto MLX (swift backup).app"
ditto "build/Flyto MLX.app" "/Applications/Flyto MLX.app"
open -a "/Applications/Flyto MLX.app"

# 3. For a clean restart (for example to reset the memory ceiling), SIGTERM
#    the serve PID: the app sees it go, waits for the PID to be gone, then
#    starts a new server.
kill -TERM <serve pid>
```

Rollback: quit the app, move the backup bundle back, `open -a` it.

---

# Flyto MLX 菜单栏 (Rust)

`Flyto MLX.app` 是菜单栏程序, 负责启动, 接管和看护本机的 `omlx.cli serve` 进程. 它替代
`apps/omlx-mac` 里的 Swift 版, 沿用同一个 bundle id (`com.panwudi.flyto-mlx`), 安装路径,
base path 解析方式和日志位置. Swift 版先留在仓库里, 等这一版在 m5max 上跑稳一段时间再删.

和 Swift 版的不同:

- 端口上已经有健康的 fmlx 在跑 (比如 app 被杀后重开), 直接接管, 不再报端口冲突.
- 进程死活看 PID, 不看端口. 停止时先发 SIGTERM, 最多等 120 秒让进程彻底退出 (它要先卸载模型,
  把缓存写盘), 之后才 SIGKILL. 同一个 base path 的旧进程没退干净之前绝不起新进程, 免得新进程把
  旧进程占的内存算成外部占用, 把自己的内存上限压低.
- 服务进程不是从菜单停掉却没了, 就分别隔 5, 10, 20 秒重启; 连续 3 次都失败就不再自动重启, 等服务稳定跑满一分钟后才重新计数.
- 有请求在处理时, 停止, 重启, 退出都会先确认.
- 菜单里能看到已加载模型, 处理中和排队的请求数, 模型内存.
- 菜单文字按 macOS 首选语言显示中文或英文.

文件: `src/supervisor.rs` (生命周期), `src/procs.rs` (PID 和端口), `src/api.rs`
(`/health`, `/admin/api/stats`), `src/main.rs` (菜单), `assets/menubar.svg` (菜单栏图标),
`assets/app-icon.svg` (app 图标, 和管理界面的 favicon 一致). 事件日志:
`~/Library/Application Support/oMLX/logs/menubar.log`, 服务日志在同目录 `server.log`.

构建, 测试和安装的命令见上面英文部分. 安装要点: 先停用旧看门狗 (它在服务不健康时强杀 app, 正常重启时也会误伤);
旧 app 改名留作回退; 新 app 会接管还在跑的服务; 需要干净重启时对服务进程发 SIGTERM, app 会等旧进程彻底退出再起新的.
安装会让服务停一分钟左右, 动手前先通知使用的人.
