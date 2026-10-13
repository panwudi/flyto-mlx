class FlytoMlx < Formula
  desc "Flyto MLX — Apple Silicon LLM server with audio chat, DFlash, and Chinese model presets (fork of oMLX)"
  homepage "https://github.com/panwudi/flyto-mlx"
  url "https://github.com/panwudi/flyto-mlx/archive/refs/tags/v0.7.0.tar.gz"
  sha256 "c39ade8585a9073d445ca59fcb7a27bfe90e7da19a4d94ce6b4e2ed5751e6292"
  license "Apache-2.0"

  head "https://github.com/panwudi/flyto-mlx.git", branch: "main"

  depends_on "rust" => :build
  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64

  # mlx-audio pins mlx-lm==0.31.1 which conflicts with our git-pinned mlx-lm.
  # Fetch source separately so we can patch the pin before install.
  resource "mlx-audio" do
    url "https://github.com/Blaizzy/mlx-audio.git",
      revision: "51753266e0a4f766fd5e6fbc46652224efc23981"
  end

  service do
    run [opt_bin/"fmlx", "serve"]
    keep_alive true
    working_dir var
    log_path var/"log/flyto-mlx.log"
    error_log_path var/"log/flyto-mlx.log"
    environment_variables PATH: std_service_path_env
  end

  def install
    # Create venv with pip so dependency resolution works properly
    system "python3.12", "-m", "venv", libexec

    # Build Rust-based packages from source with headerpad to prevent
    # Homebrew dylib ID fixup failure (Mach-O header too small for absolute paths).
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"

    # Install flyto-mlx. xgrammar (structured output) is a core dependency.
    system libexec/"bin/pip", "install", "--no-binary", "pydantic-core,rpds-py,tiktoken", buildpath.to_s

    # Install mlx-audio with patched mlx-lm pin to avoid version conflict
    resource("mlx-audio").stage do
      inreplace "pyproject.toml", '"mlx-lm==0.31.1"', '"mlx-lm>=0.31.1"'
      system libexec/"bin/pip", "install", ".[all]"
    end

    # python-multipart is declared in flyto-mlx's [audio] extra, not in mlx-audio
    system libexec/"bin/pip", "install", "python-multipart>=0.0.5"

    # Install both CLI names:
    #   fmlx — Flyto MLX brand primary
    #   omlx — backward-compat alias for users migrating from upstream oMLX
    bin.install_symlink Dir[libexec/"bin/fmlx"]
    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  # Patch the macOS arm64 xgrammar wheel so its native binding loads: after
  # Homebrew's dylib fixup the binding no longer finds @rpath/libtvm_ffi.dylib
  # (jundot/omlx#1005). xgrammar is a core dependency, so this always runs.
  def post_install
    ohai "Patching xgrammar macOS arm64 wheel"
    py = libexec/"bin/python"
    site = Utils.safe_popen_read(py, "-c",
                                 "import site; print(site.getsitepackages()[0])").chomp
    tvmlib = Utils.safe_popen_read(py, "-c",
      "import os, tvm_ffi; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib'))").chomp
    dylib = "#{site}/xgrammar/libxgrammar_bindings.dylib"
    dist_dirs = Dir["#{site}/xgrammar-*.dist-info"]

    ohai "  site=#{site}"
    ohai "  tvmlib=#{tvmlib}"
    ohai "  dylib=#{dylib} (exists? #{File.exist?(dylib)})"
    ohai "  dist-info=#{dist_dirs.inspect}"

    odie "xgrammar dylib not found at #{dylib}" unless File.exist?(dylib)
    odie "xgrammar dist-info not found under #{site}" if dist_dirs.empty?

    rpaths = Utils.safe_popen_read("/usr/bin/otool", "-l", dylib)
    if rpaths.include?(tvmlib)
      ohai "  rpath already points at tvm_ffi/lib"
    else
      ohai "  adding rpath -> #{tvmlib}"
      system "/usr/bin/install_name_tool", "-add_rpath", tvmlib, dylib
      system "/usr/bin/codesign", "--force", "--sign", "-", dylib
    end

    record = "#{dist_dirs.first}/RECORD"
    if File.exist?(record) && File.read(record).include?("libxgrammar_bindings.dylib")
      ohai "  RECORD already lists the dylib"
    else
      ohai "  writing dylib entry to #{record}"
      File.open(record, "a") { |f| f.puts "xgrammar/libxgrammar_bindings.dylib,," }
    end

    ohai "  verifying import xgrammar..."
    system py, "-c", "import xgrammar; print('xgrammar import OK')"
  end

  test do
    assert_match version.to_s,
      shell_output("#{libexec}/bin/python -c 'import omlx._version as v; print(v.__version__)'")
    system libexec/"bin/python", "-c", "import xgrammar, mlx.core, omlx.server"
    system bin/"fmlx", "--help"
  end
end
