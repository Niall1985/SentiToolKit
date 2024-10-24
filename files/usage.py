from sentitoolkit.Main_tensor_model import SentiToolKit

model = SentiToolKit()
rev = "I don't like this product"

print(model.__call__(rev))