from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("model.h5")

class_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise',
    'monkshood', 'globe thistle', 'snapdragon', "colt's foot",
    'king protea', 'spear thistle', 'yellow iris', 'globe-flower',
    'purple coneflower', 'peruvian lily', 'ball moss', 'foxglove',
    'bougainvillea', 'camellia', 'mallow', 'mexican petunia',
    'bromelia', 'blanket flower', 'trumpet creeper', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
    'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple',
    'morning glory', 'passion flower', 'lotus', 'toad lily',
    'anthurium', 'frangipani', 'clematis', 'hibiscus',
    'columbine', 'desert-rose', 'tree mallow', 'magnolia',
    'cyclamen', 'watercress', 'canna lily', 'hippeastrum',
    'bee balm', 'balloon flower', 'giant white arum lily', 'fire lily',
    'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
    'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke',
    'sweet william', 'carnation', 'garden phlox', 'love in the mist',
    'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
    'great masterwort', 'siam tulip', 'lenten rose', 'barberton daisy',
    'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
    'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
    'common dandelion', 'petunia', 'wild pansy', 'primula',
    'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
    'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata',
    'japanese anemone', 'black tulip', 'wild rose', 'orange lily',
    'gazania', 'azalea'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    label_index = np.argmax(preds)
    confidence = round(float(preds[label_index]) * 100, 2)
    label = class_names[label_index]
    return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
