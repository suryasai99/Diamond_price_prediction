from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline



application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            table = float(request.form.get('table')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        

        convert_data = data.get_data_as_dataframe()
        result = PredictPipeline().predict(convert_data)

        return render_template('results.html',final_result = result)


if __name__=='__main__':
    app.run(host = "0.0.0.0",port=5560)
