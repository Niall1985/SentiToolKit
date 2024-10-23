from sentitoolkit.Main_tensor_model import SentiToolKit

model = SentiToolKit()
rev = "I love this product"

print(model.predict_sentiment(rev))