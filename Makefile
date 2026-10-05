# Adaptive Voter Model — manuscript reproduction
#
# Typical use:
#   1. ./setup.sh                      # create .venv and install pinned deps
#   2. python scripts/download_data.py --out data   # fetch precomputed results (Zenodo)
#   3. make figures                    # regenerate all manuscript figures
#
# To regenerate the underlying data from scratch (expensive):
#   make sweep-ER sweep-WS sweep-BA
#
PYTHON ?= python
DATA_ER := data/Sweep_ER_Undirected_Magnetization
DATA_WS := data/Sweep_WS_Undirected_Magnetization
DATA_BA := data/Sweep_BA_Undirected_Magnetization

.PHONY: help test smoke sweep-ER sweep-WS sweep-BA figures figures-ER figures-WS figures-BA clean

help:
	@echo "make test         - run the pytest smoke test"
	@echo "make smoke        - quick end-to-end sweep + figure (tmp dir)"
	@echo "make sweep-<NET>  - regenerate pkl sweep for ER/WS/BA (long)"
	@echo "make figures      - regenerate all manuscript figures"
	@echo "make figures-<NET>- regenerate figures for one topology"

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/run_sweep.py --quick --graph-type ER --out /tmp/avm_smoke
	$(PYTHON) scripts/make_figures.py --data-dir /tmp/avm_smoke --network ER \
		--figures homophily --out /tmp/avm_smoke/figures

# --- data generation (expensive) -------------------------------------------------
sweep-ER:
	$(PYTHON) scripts/run_sweep.py --graph-type ER --out $(DATA_ER)
sweep-WS:
	$(PYTHON) scripts/run_sweep.py --graph-type WS --out $(DATA_WS) \
		--homo-list 1.0 0.9 0.7 0.5
sweep-BA:
	$(PYTHON) scripts/run_sweep.py --graph-type BA --out $(DATA_BA)

# --- figures ---------------------------------------------------------------------
figures: figures-ER figures-WS figures-BA

figures-ER:
	$(PYTHON) scripts/make_figures.py --data-dir $(DATA_ER) --network ER \
		--figures all --out figures

figures-WS:
	$(PYTHON) scripts/make_figures.py --data-dir $(DATA_WS) --network WS \
		--figures all --out figures

figures-BA:
	$(PYTHON) scripts/make_figures.py --data-dir $(DATA_BA) --network BA \
		--figures all --out figures

clean:
	rm -rf /tmp/avm_smoke figures/_generated .pytest_cache
