import os

directory = "c:/Users/Jakob/Projects/v2e_vonenet/cifar10dvs/airplane"

files_in_directory = os.listdir(directory)
filtered_files = [file for file in files_in_directory if file.endswith(".png")]
for file in filtered_files:
	path_to_file = os.path.join(directory, file)
	os.remove(path_to_file)