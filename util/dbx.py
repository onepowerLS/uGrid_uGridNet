import os
import shutil
import dropbox
from dotenv import load_dotenv
from dropbox.exceptions import ApiError

from constants import OUTPUT_FOLDER, DROPBOX_FOLDER, CONTRACTS_FOLDER, DROPBOX_CONTRACTS_FOLDER
from util import get_url

load_dotenv()

token = os.environ.get('DROPBOX_TOKEN')
# print(os.environ)
# token = "ttRHE4dO5-MAAAAAAAAAAUiwIG3amAe-Q8eelxJsve67yII-xE58wCdnMvP0KlYz"
DBX = dropbox.Dropbox(oauth2_access_token=token)


def upload_contracts_via_api(contracts_list):
    print("Uploading contracts to Dropbox...")
    # the source file
    urls = []
    for filename in contracts_list:
        filepath = CONTRACTS_FOLDER / filename  # path object, defining the file
        # open the file and upload it
        target_name = DROPBOX_CONTRACTS_FOLDER + filename
        with filepath.open("rb") as f:
            # upload gives you metadata about the file
            # we want to overwite any previous version of the file
            meta = DBX.files_upload(f.read(), target_name,
                                    mode=dropbox.files.WriteMode("overwrite"))
        # create a shared link
        try:
            link = DBX.sharing_create_shared_link_with_settings(target_name)
            urls.append(link.url)
        except ApiError as api_error:
            error_as_string = str(api_error.args[1])
            link = get_url(error_as_string)[0]
            urls.append(link)
    return urls


def upload_contracts_via_local_folder(contract_list):
    print("Uploading contracts to Dropbox...")
    urls = []
    for contract in contract_list:
        source_path = OUTPUT_FOLDER / contract
        destination_path = DROPBOX_FOLDER + "/" + contract
        shutil.copyfile(source_path, destination_path)
        urls.append(destination_path)
    return urls


def get_contracts_list():
    files = DBX.files_list_folder(DROPBOX_CONTRACTS_FOLDER)
    return files.entries


def get_data(file_list):
    list_with_dets = []
    for file in file_list:
        link = DBX.sharing_list_shared_links(path=file.path_display).links[1].url
        file_name = file.name
        file_name_as_list = file_name.split("_")
        first_name = file_name_as_list[3]
        last_name = file_name_as_list[2]
        site_number = file_name_as_list[1]
        language = file_name_as_list[6][:2]
        dets = [site_number, first_name, last_name, language, link]
        list_with_dets.append(dets)
    with open('output.csv', 'w') as outfile:
        outfile.writelines(','.join(str(j) for j in i) + '\n' for i in list_with_dets)
    return list_with_dets


def upload()

