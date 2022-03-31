import sys
import time

from aws_util import create_instance, stop_instance, terminate_instance, get_running_instances
from dropbox_util import get_list_from_folder
from ssh_to_aws import run_command_on_ec2, upload_files_to_ec2


def run_one_instance(site_name, shortened_id):
    dropbox_folder = f"{CONCESSION_FOLDER}/{site_name}".lower()

    instance_id, instance_ip = get_instance_info(site_name)

    files = ["./dropbox_util.py", "./uGridNet_Revised_Optimized_Version_V3.py", "./requirements.txt", "./.env"]
    upload_files_to_ec2(files=files, instance_ip=instance_ip)

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
        f"python3 -c 'from dropbox_util import upload_output_files; upload_output_files(\"{dropbox_folder}\")'",
        # "cat log.txt"
    ]
    run_command_on_ec2(commands=commands, instance_ip=instance_ip)
    stop_instance(instance_id)
    terminate_instance(instance_id)


def run_many(village_list):
    # village_list = village_list[6:16]
    village_list = village_list[::-1][3:]
    print(village_list)
    for site_name in village_list:
        print("========================================================================================================")
        print(f"\nRunning for {site_name}\n")
        shortened_id = site_name[3:7]
        run_one_instance(site_name=site_name, shortened_id=shortened_id)


def run_concession(concession, concession_folder):
    print(concession_folder)
    village_list = [v for v in get_list_from_folder(concession_folder) if v[:3] == concession]
    village_list = [v for v in village_list if
                    v not in [
                        "RIB_01_Khubetsoana",
                        "RIB_02_Khakhathane",
                        "RIB_03_Ha_Mokena",
                        "RIB_04_Boiketlo",
                        "RIB_05_Ha_Majela",
                        "RIB_06_Sekoting_Sa_Lifariki",
                        "RIB_07_Ha_Sekhaila",
                        "RIB_09_Ha_Shekoe",
                        "RIB_10_Makatseng",
                        "RIB_13_Ha_Joele",
                        "RIB_14_Ha_Pitso",
                        "RIB_15_Ha_Ntake",
                        "RIB_16_Ha_Rantsoetsa",
                        "RIB_18_Ha_Nketsi",
                        "RIB_19_Mabusetsa",
                        "RIB_20_Ha_Ramoetsana",
                        "RIB_21_Qhojoa",
                        "RIB_22_Moeaneng_Ha_Mateu",
                        "RIB_23_Moeaneng",
                        "RIB_24_Ha_Mmantolo",
                        "RIB_25_Thaba_Sekoka",
                        "RIB_26_Ha_Sehloho",
                        "RIB_27_Ha_Mokhomo",
                        "RIB_28_Ha_Motsamai",
                        "RIB_31_Mahlatheng",
                        "RIB_33_Mankoaneng",
                        "RIB_34_Kelepeng",
                        "RIB_35_Ha_Ratsiu",
                        "RIB_36_Ha_Berente",
                        "RIB_37_Ha_Mokhethi",
                        "RIB_38_Ha_Mohlakola",
                        "RIB_40_Ha_Mabekebeke",
                        "RIB_42_Ha_Monaheng",
                        "RIB_43_Ha_Beu",
                        "RIB_44_Ha_Mpakoba",
                        "RIB_45_Ha_Nkeo",
                        "RIB_46_Ha_Mphatsi",
                        "RIB_47_Ha_Shekeshe",
                        "RIB_48_Ha_Siimane",
                        "RIB_49_Ha_Rammuso",
                        "RIB_50_Ha_Seeiso",
                        "RIB_52_Ha_Bukana",
                        "RIB_53_Ha_Lebona_Khohlong",
                        "RIB_54_Mohlanapeng",
                        "RIB_55_Thaka_Banna",
                        "RIB_56_Ha_Fako",
                        "RIB_57_Motloang",
                        "RIB_59_Masefata",
                        "RIB_61_Ha_Nthota",
                        "RIB_62_Ha_Makoae",
                        "RIB_63_Ha_Salae",
                        "RIB_64_Ha_Lebona",
                        "RIB_65_Ha_Sekhaoli",

                        "TLH_01_Motlomo",
                        "TLH_02_Bokhina",
                        "TLH_05_Patising",
                        "TLH_07_Moeaneng",
                        "TLH_08_Taung",
                        "TLH_10_Lepatsong",
                        "TLH_11_Sixteen",
                        "TLH_21_Likoekoeng",
                        "TLH_23_Mafulane",
                        "TLH_24_Likhang",

                        "KET_01_Ha_Tjotjela",
                        "KET_02_Riverside",
                        "KET_03_Ha_Nthamaha",
                        "KET_04_Ha_Rantoetse",
                        "KET_06_Ha_Thetsinyana",
                        "KET_05_Mafeteke",
                        "KET_10_Ha_Hobeng",
                        "KET_12_Ha_Khojana",
                        "KET_13_Ha_Masupha",
                        "KET_14_Sekhutlong",
                        "KET_15_Mafikeng",
                        "KET_17_Boiketlo",

                        "MAT_01_Sekhutlong",
                        "MAT_02_Linareng",
                        "MAT_03_Ha_Makau",
                        "MAT_04_Ha_Mpiti",
                        "MAT_05_Liphakoeng",
                        "MAT_06_Ha_Mokone",
                        "MAT_07_Setefane",
                        "MAT_08_Liraoeleng",
                        "MAT_09_Makhapung",
                        "MAT_10_Masuaneng",
                        "MAT_11_Mathakeng",
                        "MAT_12_Ha_Molibeli",
                        "MAT KMLs",

                        "TOS_02_Tosing",
                        "TOS_03_Ha_Mabele_A_Tlala",
                        "TOS_04_Mateleng",
                        "TOS_05_Lekhalong",
                        "TOS_06_Malumeleng",
                        "TOS_08_Khorong",
                        "TOS_07_Tereseng",
                        "TOS_11_Selomong",
                        "TOS_12_Mpapa",
                        "TOS_13_Mofokeng",
                        "TOS_16_Ha Ralebona",
                        "TOS_17_Nkobolong",
                        "TOS_18_Tiping",
                        "TOS_21_Ha_Maleka",
                        "TOS_22_Sekolong",
                        "TOS_23_Ha_Liphapang",
                        "TOS_24_Khobololo",
                        "TOS_25_Thepung",

                        "SEB_01_Ha_Kautu",
                        # "SEB_07_Moseneke",
                        "SEB_11_Kepeng",


                        "LEB_01_Mohlanapeng",
                        "LEB_04_Ha_Molomo",
                        "LEB_08_Ha_Ramokakatlela",
                    ]]

    if len(village_list) == 0:
        return
    instance_ip, instance_id = get_instance_info(concession)

    set_up_commands = [
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
    ]

    files = ["./dropbox_util.py", "./uGridNet_Revised_Optimized_Version_V3.py", "./requirements.txt", "./.env"]
    upload_files_to_ec2(files=files, instance_ip=instance_ip)

    run_command_on_ec2(commands=set_up_commands, instance_ip=instance_ip)

    for village in village_list:
        run_one_village_in_instance(village=village,concession_folder=concession_folder, instance_ip=instance_ip)


def get_instance_info(instance_name):
    instance_id = create_instance(instance_name)
    time.sleep(30)
    instance_list = get_running_instances()
    filtered = []
    for i in instance_list:
        if instance_id == i[0]:
            filtered.append(i[2])
    instance_ip = filtered[0]
    return instance_ip, instance_id


def run_one_village_in_instance(village, concession_folder, instance_ip):
    print("=======================================================================================================")
    print(f"\nRunning for {village}\n")
    dropbox_folder = f"{concession_folder}/{village}".lower()
    village_run_commands = [
        f"mkdir {village}",
        f"echo $PWD",
        f"cp dropbox_util.py uGridNet_Revised_Optimized_Version_V3.py requirements.txt .env {village}",
        f"cd {village}; ls",
        f"cd {village}/; "
        f"echo $PWD;",
        f"python3 -c 'from dropbox_util import download_input_files; download_input_files(\"{dropbox_folder}\","
        f"\"./{village}\")';",
        f"ls $PWD",
        f"cd {village}/; "
        f"python3 uGridNet_Revised_Optimized_Version_V3.py {village} {village[3:7]};",
        f"cd {village}/; "
        f"python3 -c 'from dropbox_util import upload_output_files; upload_output_files(\"{dropbox_folder}\","
        f"\"./\")'"
    ]
    run_command_on_ec2(commands=village_run_commands, instance_ip=instance_ip)


if __name__ == '__main__':
    CONCESSION = sys.argv[2]
    NUMBER = sys.argv[1]
    CONCESSION_FOLDER = f"/{NUMBER} 1PWR {CONCESSION}/(0) 1PWR {CONCESSION} WBS/(3) Engineering Design and " \
                        f"Planning/3.4. Detailed Reticulation Design/uGridNET {CONCESSION}/uGridNET {CONCESSION}"
    # villages = [v for v in get_list_from_folder(CONCESSION_FOLDER) if v[:3] == CONCESSION]
    run_concession(CONCESSION, CONCESSION_FOLDER)
    # run_one_village_in_instance(village="RIB_23_Moeaneng", concession_folder=CONCESSION_FOLDER,
    # instance_ip="13.245.32.157") create_key_pair() run_many(villages) instance_ips = ["13.244.166.238",
    # "13.245.170.25", "13.245.182.44", "13.245.33.67", "13.244.108.76", "13.245.128.188"] upload_file_to_ec2(
    # file="./.env", instance_ip="13.244.167.248") for instance_ip in instance_ips: upload_file_to_ec2(
    # file="./dropbox_util.py", instance_ip=instance_ip)
