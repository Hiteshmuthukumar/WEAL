from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import io
import base64
from supabase import create_client, Client

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="haemoscan.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Supabase if env vars are present
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["eye_image"]

    # Read image into memory
    img_bytes = file.read()
    
    # Create Base64 string for displaying on the result page
    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = file.mimetype if file.mimetype else 'image/jpeg'
    data_uri = f"data:{mime_type};base64,{encoded_img}"

    # Load image for processing
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224,224))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_tmp = img_arr.reshape(1,224,224,3)
    
    # TFLite inference
    interpreter.set_tensor(input_details[0]['index'], img_tmp)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    hb_value = round(float(prediction[0][0]),1)

    # Hb status
    if hb_value < 8:
        status = "Severe Anemia"
    elif hb_value < 11:
        status = "Mild Anemia"
    else:
        status = "Normal"

    # Save to Supabase
    if supabase:
        try:
            data, count = supabase.table('predictions').insert({
                "hb_value": hb_value,
                "status": status
            }).execute()
            print(f"Inserted into Supabase: {data}")
        except Exception as e:
            print(f"Supabase error: {e}")

    return render_template(
        "result.html",
        hb=hb_value,
        status=status,
        image_path=data_uri
    )

if __name__ == "__main__":
    app.run(debug=True)