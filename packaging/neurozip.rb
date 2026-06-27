class Neurozip < Formula
  include Language::Python::Virtualenv

  desc "Task-aware neural compression for brain/biomedical signals"
  homepage "https://github.com/xerneas3318/NeuroZip"
  url "https://github.com/xerneas3318/NeuroZip/releases/download/v0.1.0-beta.1/neurozip-0.1.0-beta.1.tar.gz"
  sha256 "83a53e1b79c3c09183adc47b5c06a7e41dc27a31a5bb5426b6e4d6ab09ec3af9"
  license "MIT"
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
