build:
	docker build -t nickosipov/production-code-lesson:latest .
	docker push nickosipov/production-code-lesson:latest

run:
	docker run --rm -p 5000:5000 --name production-code-lesson nickosipov/production-code-lesson:latest

