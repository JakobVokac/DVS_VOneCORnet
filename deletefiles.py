import os

# Used for deleting specific file types from bad preprocessing

base_dir = "d:/Datasets/cifar10dvs3slice/train/"
directories = [base_dir + label for label in  ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]]
for directory in directories:
	files_in_directory = os.listdir(directory)
	filtered_files = [file for file in files_in_directory if file.endswith(".npz")]
	for file in filtered_files:
		path_to_file = os.path.join(directory, file)
		os.remove(path_to_file)

base_dir = "d:/Datasets/cifar10dvs3slice/test/"
directories = [base_dir + label for label in  ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]]
for directory in directories:
	files_in_directory = os.listdir(directory)
	filtered_files = [file for file in files_in_directory if file.endswith(".npz")]
	for file in filtered_files:
		path_to_file = os.path.join(directory, file)
		os.remove(path_to_file)