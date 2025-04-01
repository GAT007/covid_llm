import requests
from pymongo import MongoClient

# URL of the FHIR server
fhir_server_url = "https://hapi.fhir.org/baseR4"

# Resource type you want to fetch (e.g., "Patient", "Observation", etc.)
resource_type = "Patient"

# MongoDB connection parameters
mongo_host = 'localhost'
mongo_port = 27017
mongo_db_name = 'fhir_data'
mongo_collection_name = resource_type.lower()  # Using resource type as collection name

# Making a GET request to fetch data
response = requests.get(f"{fhir_server_url}/{resource_type}")

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extracting JSON data from the response
    data = response.json()

    # MongoDB connection
    client = MongoClient(mongo_host, mongo_port)
    db = client[mongo_db_name]
    collection = db[mongo_collection_name]

    # Insert fetched data into MongoDB collection
    for entry in data.get("entry", []):
        resource = entry.get("resource")
        collection.insert_one(resource)

    # Close MongoDB connection
    client.close()

    print("Data inserted into MongoDB successfully.")
else:
    print("Failed to fetch data from the FHIR server.")
