Instructions to run on local : 

docker build -t technical_doctors_practicum .

docker run --rm -p 8880:8501 technical_doctors_practicum

Instructions to create the resource group, azure container repository and web app :

az login

RESOURCE_GROUP_NAME='practicum-rg'
LOCATION='eastus'

Create the resource group
az group create -n $RESOURCE_GROUP_NAME -l $LOCATION

Set the resource id variable
RESOURCE_ID=$(az group show \
  --resource-group $RESOURCE_GROUP_NAME \
  --query id \
  --output tsv)

echo $RESOURCE_ID
REGISTRY_NAME='practicumacr'
APP_SERVICE_PLAN_NAME='practicum-plan'
APP_SERVICE_NAME='technicaldoctorspracticum'

Create the Azure Container Repository and update its permissions
az acr create -g $RESOURCE_GROUP_NAME -n $REGISTRY_NAME --sku Basic --admin-enabled true
az acr login -n $REGISTRY_NAME
az acr update -n $REGISTRY_NAME --admin-enabled true
az acr credential show --name  $REGISTRY_NAME

Create the App Service Plan to provide easy access to the repository created above
az appservice plan create --name $APP_SERVICE_PLAN_NAME --resource-group $RESOURCE_GROUP_NAME --is-linux --sku B3

Rebuild the docker image and upload it onto the repository (Make sure you have atleast 15 gb of free space for this)
docker build -t $APP_SERVICE_NAME .
docker tag $APP_SERVICE_NAME $REGISTRY_NAME.azurecr.io/$APP_SERVICE_NAME
docker push $REGISTRY_NAME.azurecr.io/$APP_SERVICE_NAME

Run build online and recreate the image
az acr build --registry $REGISTRY_NAME --resource-group $RESOURCE_GROUP_NAME --image $APP_SERVICE_NAME .

Create the final web application
az webapp create -g $RESOURCE_GROUP_NAME -p $APP_SERVICE_PLAN_NAME -n $APP_SERVICE_NAME -i $REGISTRY_NAME.azurecr.io/$APP_SERVICE_NAME
