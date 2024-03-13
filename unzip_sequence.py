import os
import zipfile
import glob

# Specify the parent directory where the 1,445 folders are located
parent_directory = '/home/knuvi/Desktop/hojun/CVPR2023-VLSAT/data/3RScan/'

print("aa")
# Search for all directories within the parent directory
for folder in glob.glob(os.path.join(parent_directory, '*/')):
    # Construct the path to the expected zip file in this folder
    zip_file_path = os.path.join(folder, 'sequence.zip')
    
    # Define the directory where the contents will be extracted
    extract_to = os.path.join(folder, 'sequence')

    # Check if the zip file exists
    if os.path.exists(zip_file_path):
        # Create the target directory if it doesn't exist
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        
        # Attempt to unzip the file
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Unzipped {zip_file_path} into {extract_to} successfully.")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file_path} is a bad zip file and cannot be unzipped.")
    else:
        print(f"No sequence.zip found in {folder}")

# Replace '/path/to/your/parent/directory' with the actual path to your parent directory containing the folders.
