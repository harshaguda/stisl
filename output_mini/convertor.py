from keras.models import load_model
import json
import pickle
from keras.models import model_from_json
model = load_model("smallvggnet.model")
model_json = model.to_json
#print(model_json)
# with open("model.pickle", "wb") as json_file:
# 	pickle.dump(model_json, json_file)
#json_string = json.dump(model_json)
#print(json_string)
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
model_str = (pickle.dumps(model))
with open("model.pickle", "wb") as json_file:
	pickle.dump(model_str,json_file)
print("Saved model to disk")