import os

import boto3
import botocore.exceptions

REGION_NAME = "af-south-1"
KEY_NAME = "rsa-pair-test"


# Create security key pair
def create_key_pair():
    ec2_client = boto3.client("ec2", region_name=REGION_NAME)
    # try:
    key_pair = ec2_client.create_key_pair(KeyName=KEY_NAME, KeyType='rsa')

    private_key = key_pair["KeyMaterial"]

    # write private key to file with 400 permissions
    with os.fdopen(os.open(f"{KEY_NAME}.pem", os.O_WRONLY | os.O_CREAT, 0o400), "w+") as handle:
        handle.write(private_key)

    # # Skip if key already exists
    # except botocore.exceptions.ClientError:
    #     pass


# Create EC2 instance, Returns instance ID
def create_instance(name):
    ec2_client = boto3.client("ec2", region_name=REGION_NAME)
    instances = ec2_client.run_instances(
        # Name=name,
        ImageId="ami-030b8d2037063bab3",
        MinCount=1,
        MaxCount=1,
        InstanceType="t3.xlarge",
        KeyName=KEY_NAME,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': name
                    },
                ]
            },
        ],
    )

    instance = instances["Instances"][0]
    return instance["InstanceId"]


# Returns a list of running EC2 Instances
def get_running_instances():
    ec2_client = boto3.client("ec2", region_name=REGION_NAME)
    reservations = ec2_client.describe_instances(Filters=[
        {
            "Name": "instance-state-name",
            "Values": ["running"],
        }
    ]).get("Reservations")

    instance_list = []
    for reservation in reservations:
        for instance in reservation["Instances"]:
            instance_id = instance["InstanceId"]
            instance_type = instance["InstanceType"]
            public_ip = instance["PublicIpAddress"]
            private_ip = instance["PrivateIpAddress"]
            instance_list.append([instance_id, instance_type, public_ip, private_ip])

    return instance_list


# Stops specified instance from running
def stop_instance(instance_id):
    ec2_client = boto3.client("ec2", region_name=REGION_NAME)
    response = ec2_client.stop_instances(InstanceIds=[instance_id])
    print(response)


# Terminates specified EC2 instance from running
def terminate_instance(instance_id):
    ec2_client = boto3.client("ec2", region_name=REGION_NAME)
    response = ec2_client.terminate_instances(InstanceIds=[instance_id])
    print(response)


# Executes a list of commands on a Linux EC2 instance
def execute_commands_on_linux_instances(commands, instance_id):
    """Runs commands on remote linux instances
    :param commands: a list of strings, each one a command to execute on the instances
    :param instance_ids: a list of instance_id strings, of the instances on which to execute the command
    :return: the response from the send_command function (check the boto3 docs for ssm client.send_command() )
    """
    client = boto3.client('ssm', region_name=REGION_NAME)
    response = client.send_command(
        DocumentName="AWS-RunShellScript",  # One of AWS' preconfigured documents
        Parameters={'commands': commands},
        InstanceIds=[instance_id],
    )
    return response
