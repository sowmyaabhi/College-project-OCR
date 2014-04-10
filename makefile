build_project: TextDetection.pyx TextDetection.py
	cp TextDetection.py TextDetection.pyx
	python setupTextDetection.py build_ext --inplace
	rm -r *.c