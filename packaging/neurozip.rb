class Neurozip < Formula
  include Language::Python::Virtualenv

  desc "Task-aware neural compression for brain/biomedical signals"
  homepage "https://github.com/xerneas3318/NeuroZip"
  url "https://github.com/xerneas3318/NeuroZip/releases/download/v0.1.0-beta.2/neurozip-0.1.0-beta.2.tar.gz"
  sha256 "ecdd431839fdfc8ed3b51b7d1aa4c80d503837487445fd653f30e27b1055dd3c"
  license "PolyForm-Noncommercial-1.0.0"
  head "https://github.com/xerneas3318/NeuroZip.git", branch: "brew-testing"

  depends_on "python@3.12"

  # Core install is stdlib-only, so there are no Python resource stanzas to vendor.
  # The neural `ml` extra (torch etc.) is installed separately by the user.
  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "neurozip", shell_output("#{bin}/neurozip --version")
  end
end
