import numpy as np
from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import tensorflow as tf

# cargar modelo entrenado
model = tf.keras.models.load_model("modelo_numeros.h5")

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)

        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)

        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(28,28)
        arr = arr.reshape(1,28,28,1)

        prediction_values = model.predict(arr, batch_size=1, verbose=0)
        prediction = str(np.argmax(prediction_values))

        print("Predicción:", prediction)

        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self.wfile.write(prediction.encode())


print("Servidor corriendo en http://localhost:8000")

server = HTTPServer(('localhost',8000), SimpleHTTPRequestHandler)
server.serve_forever()