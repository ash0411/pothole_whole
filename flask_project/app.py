from flask import Flask,render_template,redirect,url_for,request,flash
from flask_bootstrap import Bootstrap
import os
import inference



app = Flask(__name__)
Bootstrap(app)
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static',uploaded_file.filename)
            uploaded_file.save(image_path)
            pred = inference.get_prediction(image_path)
            if pred == 0:
                class_name = "normal"
            else:
                class_name = "pothole detected"
            #print("CLASS name =",class_name)
            result = {
                'class_name' :class_name,
                'image_path' : image_path,
            }
            return render_template('show.html',result=result)
    return render_template('index.html') 
if __name__=='__main__':
    app.run(debug=True)