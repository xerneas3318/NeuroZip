class Neurozip < Formula
  include Language::Python::Virtualenv

  desc "Task-aware neural compression for brain/biomedical signals"
  homepage "https://github.com/xerneas3318/NeuroZip"
  url "https://github.com/xerneas3318/NeuroZip/releases/download/v0.2.0-beta.1/neurozip-0.2.0-beta.1.tar.gz"
  sha256 "dd17c4da8608f24e847ed1151ecdb787e6937431d080bef71c378d388f2556e4"
  license "PolyForm-Noncommercial-1.0.0"
  head "https://github.com/xerneas3318/NeuroZip.git", branch: "model-testing"

  depends_on "python@3.12"

  # Core install is stdlib-only (CLI + UI shell). Real inference needs the `ml` extra
  # (pip install "neurozip[ml]") and the model bundle (`neurozip download`).
  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "neurozip", shell_output("#{bin}/neurozip --version")
  end
end
