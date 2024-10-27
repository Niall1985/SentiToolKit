from sentitoolkit.Main_tensor_model import SentiToolKit

model = SentiToolKit()
rev = input("Enter: ")

print(model.__call__(rev))