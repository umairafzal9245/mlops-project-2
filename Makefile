install: 
		pip install --upgrade pip && \
		pip install -r requirements.txt

trainmodel:
		python model.py 

testmodel:
		python api2.py

testapi:
		python api.py