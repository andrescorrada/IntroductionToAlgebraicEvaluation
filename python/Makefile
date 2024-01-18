.PHONY: typehint
typehint:
	mypy --ignore-missing-imports ntqr/

.PHONY: test
test:
	pytest tests/

.PHONY: lint
lint:
	pylint ntqr/

.PHONY: checklist
checklist: lint typehint test

.PHONY: black
black:
	black -l 79 src/ntqr tests/

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr
	find . -type d -name __pycache__ | xargs rm -fr

