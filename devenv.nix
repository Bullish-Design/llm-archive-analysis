{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.PROJECT_NAME = "llm-archive-analysis";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.uv
  ];

  # https://devenv.sh/languages/
  languages = {
    python = {
      enable = true;
      version = "3.13";
      venv.enable = true;
      uv.enable = true;
    };
  };

  # https://devenv.sh/scripts/
  scripts.install.exec = ''
    echo "Installing llm-archive-analysis..."
    uv sync
  '';

  scripts.test.exec = ''
    echo "Running tests..."
    uv run pytest
  '';

  scripts.lint.exec = ''
    echo "Running linters..."
    uv run ruff check src tests
  '';

  scripts.format.exec = ''
    echo "Formatting code..."
    uv run ruff format src tests
  '';

  enterShell = ''
    echo "Welcome to $PROJECT_NAME development environment"
    echo "Python $(python --version)"
    echo ""
    echo "Available commands:"
    echo "  install - Install project dependencies"
    echo "  test    - Run test suite"
    echo "  lint    - Run linters"
    echo "  format  - Format code"
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running test suite..."
    uv run pytest -v
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.ruff.enable = true;
  # pre-commit.hooks.ruff-format.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
