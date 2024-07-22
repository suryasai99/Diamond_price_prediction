from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging


application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict_datapoint():
    if request.method =='GET':
        logging.info('got get')
        return render_template('form_1.html')
    
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            table = float(request.form.get('table')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity'),
            depth = float(request.form.get('depth')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z'))
        )
        
        convert_data = data.get_data_as_dataframe()
        result = PredictPipeline().predict(convert_data)

        return render_template('form_1.html',final_result = result)


if __name__=='__main__':
    app.run(host = "0.0.0.0",port=5580)
