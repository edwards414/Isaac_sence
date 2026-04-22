PYTHON := $(HOME)/isaac-sim/isaac-sim-standalone-4.5.0-linux-x86_64/python.sh
SCRIPT := scripts/generate_grass_dataset.py

# Default output directory for normal runs (can be overridden via --output-dir)
OUTPUT_DIR ?= output_dataset

.PHONY: run
run:
	$(PYTHON) $(SCRIPT) --output-dir $(OUTPUT_DIR)

.PHONY: run-headless
run-headless:
	$(PYTHON) $(SCRIPT) --output-dir $(OUTPUT_DIR) --headless

# ------------------------------------------------------------
# Testing / Smoke‑test targets
# ------------------------------------------------------------
# Small test configuration – generates a minimal dataset
TEST_OUTPUT_DIR ?= test_output
SMOKE_CLIPS ?= 1
SMOKE_FRAMES ?= 2

.PHONY: test-smoke-visual
# Same as test-smoke but WITH Isaac Sim UI visible
test-smoke-visual: clean-test
	@echo "Running smoke test WITH UI ($(SMOKE_CLIPS) clip, $(SMOKE_FRAMES) frames per clip)…"
	$(PYTHON) $(SCRIPT) \
		--output-dir $(TEST_OUTPUT_DIR) \
		--clips-per-scene $(SMOKE_CLIPS) \
		--frames-per-clip $(SMOKE_FRAMES) \
		--visible
	@if [ -f $(TEST_OUTPUT_DIR)/train/clip_000001/rgb/rgb_000000.png ]; then \
		echo "✅ Smoke test passed – sample output found."; \
	else \
		echo "❌ Smoke test failed – no output file detected."; exit 1; \
	fi

.PHONY: clean-test
clean-test:
	rm -rf $(TEST_OUTPUT_DIR)

.PHONY: test-smoke
# Run a tiny generation and verify a sample file exists
test-smoke: clean-test
	@echo "Running smoke test ($(SMOKE_CLIPS) clip, $(SMOKE_FRAMES) frames per clip)…"
	$(PYTHON) $(SCRIPT) \
		--output-dir $(TEST_OUTPUT_DIR) \
		--clips-per-scene $(SMOKE_CLIPS) \
		--frames-per-clip $(SMOKE_FRAMES) \
		--headless
	@# Verify that at least one RGB frame was produced
	@if [ -f $(TEST_OUTPUT_DIR)/train/clip_000001/rgb/rgb_000000.png ]; then \
		echo "✅ Smoke test passed – sample output found."; \
	else \
		echo "❌ Smoke test failed – no output file detected."; exit 1; \
	fi
