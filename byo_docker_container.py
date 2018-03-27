# Intended for use in 'byo_docker_container' notebook.
# This is a fork of scikit_bring_your_own, re-written to use boto3 SageMaker.

#Setup environment
import boto3
import re
import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
from datetime import datetime

role = get_execution_role()

######### Parameters #########
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
DockerContainer = '{}.dkr.ecr.{}.amazonaws.com/decision-trees-sample'.format(account, region)


#S3 location
bucket = 'scikit-byo-iris'
s3bucketURI = 's3://' + bucket_prefix + '/'
TrainingPrefix = 'TrainingData'
ValidationPrefix =  'ValidationData'
OuputPrefix = 'Output'

#Training Cluster params
TrainingInstanceType = ml.p3.2xlarge #'ml.m4.xlarge'|'ml.m4.4xlarge'|'ml.m4.10xlarge'|'ml.c4.xlarge'|'ml.c4.2xlarge'|'ml.c4.8xlarge'|'ml.p2.xlarge'|'ml.p2.8xlarge'|'ml.p2.16xlarge'|'ml.p3.2xlarge'|'ml.p3.8xlarge'|'ml.p3.16xlarge'|'ml.c5.xlarge'|'ml.c5.2xlarge'|'ml.c5.4xlarge'|'ml.c5.9xlarge'|'ml.c5.18xlarge'
TrainingInstanceCount = 3
TrainingInstanceVolume = 123 #Volume Size for the training instance(s), in GB. General purpose SSD (GP2)
TrainingJobName = 'byo_container_' + str(datetime.now())
hyperparams = {'test':'test'}

##### Begin training job setup ######
sage = boto3.client('sagemaker')

response = sage.CreateTrainingJob(
    TrainingJobName= TrainingJobName,
    HyperParameters= hyperparams,
    AlgorithmSpecification={
        'TrainingImage': DockerContainer,
        'TrainingInputMode': 'File'
    },
    RoleArn= role,
    InputDataConfig=[
        {
            'ChannelName': 'TrainingData',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': s3bucketURI + TrainingPrefix
                    'S3DataDistributionType': 'ShardedByS3Key'
                }
            },
            'ContentType': 'string',
        },
        {
            'ChannelName': 'ValidationData',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': s3bucketURI + ValidationPrefix
                    'S3DataDistributionType': 'ShardedByS3Key'
                }
            },
            'ContentType': 'string',
        },
    ],
    OutputDataConfig={
        'S3OutputPath': s3s3bucketURI + OuputPrefix
    },
    ResourceConfig={
        'InstanceType': TrainingInstanceType,
        'InstanceCount': TrainingInstanceCount,
        'VolumeSizeInGB': TrainingInstanceVolume,
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 200
    },
    )
