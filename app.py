import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# Initializing our app 
app=Flask(__name__)
# Open our pickle file in which model information is stored.
model_classification =pickle.load(open('Random_forest_regressor_model.pkl','rb'))
model_regresson =pickle.load(open('XG_Boost_regressor_model_Pickle_.pkl','rb'))

#This function will execute as our home screen stored in home.html
@app.route('/',methods = ['GET'])
def home():
    #return 'Hello World'
    return render_template('home.html') #here

    ################################################################333
#This function is linked with the html form by the name of predict and 
# this will first take up the value from the form and then predict it with 
# our model stored in the pickle
@app.route('/predict_Classifier',methods = ['POST'])
def predict_Classifier():
    data = [float(x) for x in request.form.values()]
    final_feature = [np.array(data)]
    row_to_predict = final_feature
    df_single_values = pd.DataFrame(row_to_predict , columns = model_classification.feature_names_in_)
    print(df_single_values)
    output = model_classification.predict(df_single_values)[0]
    print(output)
    output = round(output,2)
    return render_template('home.html', prediction_text = "Fire will be there :: {}".format(output))
####################################################333

#This is the code for the operation on the second form

@app.route('/predict_Reg',methods = ['POST'])
def predict_Reg():
    data = [float(x) for x in request.form.values()]
    final_feature = np.array(data)
    import pandas as pd

    row_to_predict = [final_feature] 
    df = pd.DataFrame(row_to_predict , columns = model_regresson.feature_names_in_)
    print(df)
    output = model_regresson.predict(df)[0]
    print(output)
    output = round(output,2)
    return render_template('home.html', prediction_text_ = "Fire will be there :: {}".format(output))

if __name__=='__main__':
    app.run(debug = True)
    

