import boto3
import botocore
import paramiko


KEY = paramiko.RSAKey.from_private_key_file("rsa-pair-test.pem")
SSH_CLIENT = paramiko.SSHClient()
SSH_CLIENT.set_missing_host_key_policy(paramiko.AutoAddPolicy())


def upload_file_to_ec2(file, instance_ip):
    transport = paramiko.Transport((instance_ip, 22))
    transport.connect(username="ubuntu", pkey=KEY)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(f"{file}", f"/home/ubuntu/{file}")


def upload_files_to_ec2(files, instance_ip):
    transport = paramiko.Transport((instance_ip, 22))
    transport.connect(username="ubuntu", pkey=KEY)

    sftp = paramiko.SFTPClient.from_transport(transport)
    for file in files:
        sftp.put(f"{file}", f"/home/ubuntu/{file}")


def run_command_on_ec2(instance_ip, commands):
    # Connect/ssh to an instance
    try:
        # Here 'ubuntu' is user name and 'instance_ip' is public IP of EC2
        SSH_CLIENT.connect(hostname=instance_ip, username="ubuntu", pkey=KEY)

        for command in commands:
            print(f"\n{command}\n")
            # Execute a command(cmd) after connecting/ssh to an instance
            stdin, stdout, stderr = SSH_CLIENT.exec_command(command)
            print(stdout.read())

        # close the client connection once the job is done
        SSH_CLIENT.close()

    except Exception as e:
        print(e)

# except Exception, e:
#     print(e)

