CONDA_ENV_DIR := ./.env
PYTHON = ${CONDA_ENV_DIR}/bin/python3

.PHONY: create-env
## Create conda env
create-env:
	@conda env create -f environment.yml -p .env

.PHONY: clean-env
## Remove conda environment
clean-env:
	@rm -rf .env

.PHONY: install-e
## Install im2gps as editable
install-e:
	@${PYTHON} -m pip install -e .

.PHONY: refresh-conda-env
## Refresh conda environment
refresh-conda-env: clean-env create-env

.PHONY: update-env
## Update conda environment
update-env:
	@conda env update -f environment.yml -p .env

.PHONY: dist-wheel
## Build wheel artifact
dist-wheel:
	@${PYTHON} setup.py check
	@${PYTHON} setup.py bdist_wheel

.PHONY: clean-dist
## Clean distribution folders
clean-dist:
	-@${PYTHON} setup.py clean --all
	-@rm -rf {build,dist,*.egg-info}

.PHONY: run-tests
## Run unit tests
run-tests:
	-@${PYTHON} -m pytest -v

.PHONY: help
# Adapted by David Prihoda from: https://raw.githubusercontent.com/nestauk/patent_analysis/3beebda/Makefile
## Auto-generated help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| less $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')