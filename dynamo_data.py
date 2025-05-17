import boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb')


def get_dynamodb_item(table_name, key_dict):
    """
    Obtiene un ítem de DynamoDB con manejo de errores.
    Args:
        table_name (str): Nombre de la tabla DynamoDB.
        key_dict (dict): Diccionario con la clave primaria (PK y SK).
    Returns:
        dict or None: El ítem encontrado o None si no existe o falla.
    """
    table = dynamodb.Table(table_name)
    try:
        response = table.get_item(Key=key_dict)
        return response.get('Item')
    except ClientError as e:
        print(f"[ERROR] DynamoDB ClientError: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
    return None


def scan_documents_by_date_and_type(table_name, date_prefix=None, doc_type=None, limit=100):
    """
    Escanea una tabla DynamoDB filtrando por fecha en documentId o tipo.
    Args:
        table_name (str): Nombre de la tabla.
        date_prefix (str): Prefijo de fecha tipo '2024/01/01' dentro del documentId.
        doc_type (str): Tipo de documento esperado ('pdf_text', 'image', etc.).
        limit (int): Límite de resultados a devolver.
    Returns:
        list[dict]: Lista de documentos que cumplen los filtros.
    """
    table = dynamodb.Table(table_name)
    try:
        scan_kwargs = {
            'Limit': limit
        }

        results = []
        done = False
        start_key = None

        while not done:
            if start_key:
                scan_kwargs['ExclusiveStartKey'] = start_key

            response = table.scan(**scan_kwargs)
            items = response.get('Items', [])

            for item in items:
                doc_id = item.get('documentId', '')
                type_ok = (doc_type is None or item.get('documentType') == doc_type)
                date_ok = (date_prefix is None or date_prefix in doc_id)
                if type_ok and date_ok:
                    results.append(item)

            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None or len(results) >= limit

        return results

    except ClientError as e:
        print(f"[ERROR] DynamoDB ClientError: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
    return []


# ? Ejemplo de uso
if __name__ == "__main__":
    # Obtener un documento específico
    item = get_dynamodb_item(
        table_name="poc-entities-dynamodb-datagob-maptech",
        key_dict={
            "documentId": "classified/2024/01/01/pdf_text/1234/invoice.pdf",
            "recordId": "op78903a-8908-3f5z-9860-6e8o9a710dfd"
        }
    )
    print("Entidad extraída:", item)

    # Escanear documentos de un día específico y tipo
    matches = scan_documents_by_date_and_type(
        table_name="poc-metadata-dynamodb-datagob-maptech",
        date_prefix="2024/01/01",
        doc_type="pdf_text",
        limit=50
    )
    print(f"Se encontraron {len(matches)} documentos:")
    for doc in matches:
        print(doc.get("documentId"))
