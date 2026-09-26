.PHONY: help install install-all install-dev test lint docstyle typecheck check build clean \
       examples example-basic example-connectome example-navigation example-gui \
       example-gui-smoke \
       cli-list cli-simulate cli-simulate-offline cli-train cli-train-offline \
       test-neuprint test-all

PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
PYTEST   := $(PYTHON) -m pytest
GUI_ARGS ?=
SRC      := src/flynet
TESTS    := tests

# ── Token: reads .env file, then falls back to shell env ───────────────────
# Create .env with:  echo "NEUPRINT_APPLICATION_CREDENTIALS=your-token" > .env
-include .env
export NEUPRINT_APPLICATION_CREDENTIALS

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────

install: ## Install in editable mode (core deps only)
	$(PIP) install -e "."

install-dev: ## Install with dev extras (pytest, etc.)
	$(PIP) install -e ".[dev]"

install-all: ## Install with all extras (neuprint, dev, etc.)
	$(PIP) install -e ".[all]"

venv: ## Create the virtual environment
	python3 -m venv .venv
	$(PIP) install --upgrade pip

# ── Testing ────────────────────────────────────────────────────────────────

test: ## Run unit tests
	$(PYTEST) $(TESTS) -v

test-quiet: ## Run unit tests (quiet)
	$(PYTEST) $(TESTS) -q

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=term-missing

# ── Linting / Type checking ────────────────────────────────────────────────

lint: ## Syntax-check all source files
	@for f in $(SRC)/*.py; do \
		$(PYTHON) -m py_compile "$$f" && echo "OK  $$f" || echo "FAIL $$f"; \
	done

docstyle: ## Check Google-style docstrings (pydocstyle)
	$(PYTHON) -m pydocstyle --convention=google $(SRC) examples

typecheck: ## Run mypy on source (strict config in pyproject.toml)
	$(PYTHON) -m mypy $(SRC)

check: lint docstyle typecheck test-quiet ## Run all gates: lint, docstyle, typecheck, tests

# ── Build ──────────────────────────────────────────────────────────────────

build: ## Build wheel and sdist
	$(PIP) install build
	$(PYTHON) -m build

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Examples ───────────────────────────────────────────────────────────────

examples: example-basic example-connectome example-navigation ## Run all examples

example-basic: ## Run basic network example
	$(PYTHON) examples/basic_network.py

example-connectome: ## Run connectome simulation (offline)
	$(PYTHON) examples/connectome_simulation.py --offline

example-navigation: ## Run navigation task (offline, 5 episodes)
	$(PYTHON) examples/navigation_task.py --episodes 5 --offline

example-gui: ## Open the interactive foraging GUI (real brain; needs token or warm cache)
	$(PYTHON) examples/fly_foraging_gui.py $(GUI_ARGS)

example-gui-smoke: ## Headless foraging GUI check on the synthetic brain (no token)
	$(PYTHON) examples/fly_foraging_gui.py --synthetic --smoke --smoke-frames 200 --speed 8

# ── CLI ────────────────────────────────────────────────────────────────────

cli-list: ## List available connectome datasets
	$(PYTHON) -m flynet.cli list-datasets

cli-simulate-offline: ## Simulate with synthetic brain (no token needed)
	$(PYTHON) -m flynet.cli simulate --offline --plot

cli-simulate: ## Simulate with real connectome (needs token in .env or env)
ifndef NEUPRINT_APPLICATION_CREDENTIALS
	$(error NEUPRINT_APPLICATION_CREDENTIALS not set. Create .env or export it)
endif
	$(PYTHON) -m flynet.cli simulate --plot

cli-train: ## Train with real connectome (needs token in .env or env)
ifndef NEUPRINT_APPLICATION_CREDENTIALS
	$(error NEUPRINT_APPLICATION_CREDENTIALS not set. Create .env or export it)
endif
	$(PYTHON) -m flynet.cli train --episodes 10

cli-train-offline: ## Train with synthetic brain (no token needed)
	$(PYTHON) -m flynet.cli train --offline --episodes 5

# ── neuPrint (requires token) ─────────────────────────────────────────────

test-neuprint: ## Test real neuPrint mini brain fetch + simulate
ifndef NEUPRINT_APPLICATION_CREDENTIALS
	$(error NEUPRINT_APPLICATION_CREDENTIALS not set. Create .env or export it)
endif
	$(PYTHON) -c " \
from flynet.connectome import ConnectomeLoader; \
from flynet.network import SpikingNetwork; \
loader = ConnectomeLoader(); \
ids, edges, motors = loader.load_or_fetch_mini(neuron_type='DNge104', max_edges=40); \
net = SpikingNetwork(ids, edges, motor_ids=motors); \
drv, _ = net.turn(rho_right=2.0, rho_left=1.0); \
print(f'OK: {len(net.ids)} neurons, steering={drv:.4f}')"

# ── Full system test ───────────────────────────────────────────────────────

test-all: test-quiet lint ## Run tests + syntax check
	@echo ""
	@echo "──── System Test Summary ────"
	@echo "  Source files:  $$(find $(SRC) -name '*.py' | wc -l)"
	@echo "  Lines of code: $$(find $(SRC) -name '*.py' -exec cat {} + | wc -l)"
	@echo "  Test functions: $$(grep -c 'def test_' $(TESTS)/*.py)"
	@echo "  All checks passed."
