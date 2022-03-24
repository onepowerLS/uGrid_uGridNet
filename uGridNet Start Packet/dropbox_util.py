import os
import re
import shutil
from pathlib import Path

import dropbox
from dotenv import load_dotenv
from dropbox.exceptions import ApiError

# from uGridNet_Revised_Optimized_Version_V3_1 import FOLDER

load_dotenv()

token = os.environ.get('DROPBOX_TOKEN')
# print(os.environ)
# token = "ttRHE4dO5-MAAAAAAAAAAUiwIG3amAe-Q8eelxJsve67yII-xE58wCdnMvP0KlYz"
DBX = dropbox.Dropbox(oauth2_access_token=token)


def upload_file_to_dropbox(filepath, destination_path):
    # the source file
    # open the file and upload it
    with filepath.open("rb") as f:
        # upload gives you metadata about the file
        # we want to overwite any previous version of the file
        DBX.files_upload(f.read(), destination_path, mode=dropbox.files.WriteMode("overwrite"))
    # create a shared link


def get_url(string_input):
    return re.findall(r'(https?://\S+)', string_input)


def get_list_from_folder(folder):
    files = DBX.files_list_folder(folder)
    return [f.name for f in files.entries]


def download_file_from_dropbox(filename, folder, filepath=None):
    if filepath is None:
        if filename is None or folder is None:
            return None
        else:
            filepath = '{}/{}'.format(folder, filename)
    output_path = "{}".format(filename)
    DBX.files_download_to_file(path=filepath.lower(), download_path=output_path)
    # metadata = DBX.files_download_to_file(path=filepath, download_path=output_path)
    return output_path


def is_output_file(f):
    result = f[-4:] in ["1.py"] or f[-4:] in [".kml", ".pdf", ".csv"] or f[-5:] in [".xlsx"]
    return result


def upload_output_files(dropbox_folder):
    local_folder = "."
    exclude = ["venv", ".cache", ".config", ".local", ".ssh", "__pycache__", ".idea"]
    for root, dirs, files in os.walk(local_folder, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        files = [f for f in files if is_output_file(f)]
        for filename in files:
            # construct the full local path
            local_path = Path(os.path.join(root, filename))

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_folder)
            dropbox_path = os.path.join(dropbox_folder, relative_path)

            upload_file_to_dropbox(filepath=local_path, destination_path=dropbox_path)


def download_input_files(dropbox_folder):
    files = get_list_from_folder(dropbox_folder)
    files = [f for f in files if "." in f]
    for f in files:
        print(f)
        download_file_from_dropbox(filename=f, folder=dropbox_folder)
