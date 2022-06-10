docs: ## Render annotated HTML
	find ./docs/ -name "*.html" -type f -delete
	find ./docs/ -name "*.svg" -type f -delete
	pylit --remove_empty_sections --title_md -t ../../pylit/templates/research -d docs -w *

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean build check upload help docs
.DEFAULT_GOAL := help
