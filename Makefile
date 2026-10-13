.DEFAULT_GOAL := default

reinstall_package:
	@pip uninstall -y lewagon || :
	@pip install -e .

run_api:
	uvicorn api.app:app --reload

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes
