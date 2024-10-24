from sentitoolkit.Main_tensor_model import SentiToolKit

model = SentiToolKit()
rev = "I dislike this product"

print(model.__call__(rev))