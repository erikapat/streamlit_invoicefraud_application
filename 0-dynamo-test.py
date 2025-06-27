from dotenv import load_dotenv
import os

# Cargar las variables desde .env
load_dotenv()

import boto3

# Crear conexión con DynamoDB usando las variables cargadas
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_DEFAULT_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

for table in dynamodb.tables.all():
    print("Tabla:", table.name)
