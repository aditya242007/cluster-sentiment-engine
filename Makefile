.PHONY: setup generate pipeline dashboard test figures clean

setup:
	pip install -r requirements.txt

generate:
	PYTHONPATH=. python3 -m src.cri.generate

pipeline:
	PYTHONPATH=. python3 -c "from src.cri.generate import generate_synthetic_reviews; from src.cri.normalize import normalize_text; from src.cri.classify import RuleBasedClassifier; from src.cri.benchmark import benchmark_property, generate_action_report; df = generate_synthetic_reviews(5000, 42); c = RuleBasedClassifier(); df['Food'] = [c.classify(normalize_text(t))['Food']['score'] for t in df['review_text']]; res = benchmark_property('P1', 'C1', df, 'Food'); print(generate_action_report(res))"

dashboard:
	PYTHONPATH=. streamlit run app.py

test:
	PYTHONPATH=. pytest tests/ -v --cov=src/cri

figures:
	PYTHONPATH=. python3 scripts/make_figures.py

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f figures/*.png
	rm -f data/synthetic/reviews.csv
