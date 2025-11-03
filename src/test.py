import os

directory_path = './data/weights/selected' # Use '.' for the current working directory

# os.listdir() returns a list of strings directly
file_list = os.listdir(directory_path)

print("\n--- os.listdir() Output ---")
print(file_list)