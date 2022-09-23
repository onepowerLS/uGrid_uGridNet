import time

from aws_util import create_key_pair, create_instance, stop_instance, terminate_instance, get_running_instances, \
    execute_commands_on_linux_instances
from ssh_to_aws import upload_file_to_ec2, run_command_on_ec2


def run_one_instance(site_name, shortened_id):
    dropbox_folder = f"{CONCESSION_FOLDER}/{site_name}".lower()

    instance_id = create_instance(site_name)
    time.sleep(30)
    instance_list = get_running_instances()
    filtered = []
    for i in instance_list:
        if instance_id == i[0]:
            filtered.append(i[2])
    instance_ip = filtered[0]

    upload_file_to_ec2(file="./dropbox_util.py", instance_ip=instance_ip)
    upload_file_to_ec2(file="./uGridNet_Revised_Optimized_Version_V3.py", instance_ip=instance_ip)
    upload_file_to_ec2(file="./requirements.txt", instance_ip=instance_ip)
    upload_file_to_ec2(file="./.env", instance_ip=instance_ip)
    commands = [
        "sudo apt update ",
        "sudo apt install software-properties-common -y",
        "sudo add-apt-repository ppa:deadsnakes/ppa",
        "sudo apt update",
        "sudo add-apt-repository universe",
        "sudo apt install python3 -y",
        "sudo apt install python3-pip -y",
        "sudo apt install poppler-utils -y",
        "pip install -r requirements.txt",
        "pip install --upgrade numpy",
        f"python3 -c 'from dropbox_util import download_input_files; download_input_files(\"{dropbox_folder}\")'",
        f"python3 uGridNet_Revised_Optimized_Version_V3.py {site_name} {shortened_id}",
        # f"python3 -c 'from dropbox_util import upload_output_files; upload_output_files(\"{dropbox_folder}\")'",
        # "cat log.txt"
    ]
    run_command_on_ec2(commands=commands, instance_ip=instance_ip)
    # stop_instance(instance_id)
    # terminate_instance(instance_id)


CONCESSION = "MAT"
NUMBER = "1_2"
CONCESSION_FOLDER = f"/{NUMBER} 1PWR {CONCESSION}/(0) 1PWR {CONCESSION} WBS/(3) Engineering Design and Planning/3.4. " \
                    f"Detailed Reticulation " \
                    f"Design/uGridNET {CONCESSION}/uGridNet {CONCESSION}"
# CONCESSION_FOLDER = "/0_4 1PWR SEH/(0) 1PWR SEH WBS/(3) Engineering Design and Planning/3.4. Detailed Reticulation " \
#                     "Design/uGridNET SEH/uGridNet SEH"
villages = [
    "MAT_01_Sekhutlong",
    "MAT_02_Linareng",
    "MAT_03_Ha_Makau",
    "MAT_04_Ha_Mpiti",
    "MAT_05_Liphakoeng",
    "MAT_06_Ha_Mokone",
    "MAT_07_Setefane",
    "MAT_08_Liraoeleng",
    "MAT_09_Makhapung",
    "MAT_10_Masuoaneng",
    "MAT_11_Mathakeng"
]


def run_many(village_list):
    for site_name in village_list:
        print(f"Running for {site_name}")
        shortened_id = site_name[3:7]
        run_one_instance(site_name=site_name, shortened_id=shortened_id)


if __name__ == '__main__':
    # create_key_pair()
    run_many(villages)
    # upload_file_to_ec2(file="./.env", instance_ip="13.244.167.248")
