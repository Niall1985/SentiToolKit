from sentitoolkit.Main_tensor_model import SentiToolKit

model = SentiToolKit()
rev = "The battery life is really poor, i don't like it"

print(model.__call__(rev))