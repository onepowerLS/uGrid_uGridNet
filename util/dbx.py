import os
import re

import dropbox
from dotenv import load_dotenv
from dropbox.exceptions import ApiError

from pathlib import Path

# from util.util import get_url

load_dotenv()

token = os.environ.get('DROPBOX_TOKEN')
# print(os.environ)
# token = "ttRHE4dO5-MAAAAAAAAAAUiwIG3amAe-Q8eelxJsve67yII-xE58wCdnMvP0KlYz"
DBX = dropbox.Dropbox(oauth2_access_token=token)


def upload(origin_file_path, destination_filepath):
    origin_file_path = Path(origin_file_path)
    with origin_file_path.open("rb") as f:
        meta = DBX.files_upload(f.read(), destination_filepath, mode=dropbox.files.WriteMode("overwrite"))
        try:
            link = DBX.sharing_create_shared_link_with_settings(destination_filepath)
        except ApiError as api_error:
            error_as_string = str(api_error.args[1])
            link = get_url(error_as_string)[0]
    return link


def download(origin_file_path, destination_filepath):
    with open(destination_filepath, "wb+") as f:
        metadata, res = DBX.files_download(path=origin_file_path)
        if res is None:
            success = True
        else:
            f.write(res.content)
            success = False
    return success


def list_all(folder_path):
    response = DBX.files_list_folder(folder_path)
    return response.entries


def list_only_files(folder_path):
    pass


def list_only_directories(folder_path):
    pass


def get_url(string_input):
    return re.findall(r'(https?://\S+)', string_input)


# upload("/Users/mostation/Projects/1PWR/uGrid_uGridNet/util/util.py", "/1PWR IT/test.py")
# download("/1PWR IT/onboarding.docx", "/Users/mostation/Projects/1PWR/uGrid_uGridNet/downloaded.docx")
print(list_all("/1PWR IT/"))

files = [1,2,3]
for i in files:
    download(i, "ngd"+i)