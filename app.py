from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn



# Define the data model
class Item(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    

# Load the model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the FastAPI application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/recommend')
def recommend(item: Item):
    # Print the data received
    # print(item)
    # Make a prediction using the model
    prediction = model.predict([[17.99,10.38,122.80,1001.0,0.118]])
    
   
    # Retrieve the class label
    
    output = prediction[0]
    if output == 0:
        return {"result": "Benign"}
    else:
        return {"result": "Malignant"}
    


# Run the application using uvicorn
# This should be in a separate file or under a __name__ == "__main__" condition
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
