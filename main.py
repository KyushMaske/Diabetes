from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import numpy as np
import os
import psycopg2
import pandas as pd
from datetime import datetime
from shutil import move
from backend import DiabetesModel
from dotenv import load_dotenv
import requests
from pipeline import pipeline
import threading


load_dotenv()

app = FastAPI()


templates = Jinja2Templates(directory="templates")


DATABASE_URL = os.getenv("DATABASE_URL")


app.mount("/static", StaticFiles(directory="."), name="static")


diabetes_model = DiabetesModel("diabetes_model.pkl", "scaler.pkl", DATABASE_URL)


# Serve the HTML file at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class DiabetesData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int


# Endpoint to make a prediction
@app.post("/predict")
def predict(data: DiabetesData):
    input_data = np.array(
        [
            [
                data.pregnancies,
                data.glucose,
                data.blood_pressure,
                data.skin_thickness,
                data.insulin,
                data.bmi,
                data.diabetes_pedigree_function,
                data.age,
            ]
        ]
    )

    result = diabetes_model.predict(input_data)

    # Save the prediction in the database
    try:
        diabetes_model.save_prediction(data.dict(), result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insert error: {str(e)}")

    return {"prediction": result}


# Fetch data from the database
def fetch_all_results():
    try:

        conn = psycopg2.connect(DATABASE_URL)
        query = """
        SELECT id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, 
               diabetes_pedigree_function, age, prediction 
        FROM diabetes_predictions;
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df.to_dict(orient="records")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Route to display results in HTML
@app.get("/results", response_class=HTMLResponse)
async def get_results(request: Request):
    results = fetch_all_results()
    if results is None:
        raise HTTPException(
            status_code=500, detail="Failed to fetch data from the database"
        )

    return templates.TemplateResponse(
        "results.html", {"request": request, "results": results}
    )


# Endpoint to delete a specific record
@app.delete("/delete/{record_id}")
async def delete_record(record_id: int):
    try:

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        delete_query = "DELETE FROM diabetes_predictions WHERE id = %s"
        cursor.execute(delete_query, (record_id,))
        conn.commit()

        cursor.close()
        conn.close()

        return JSONResponse(
            status_code=200, content={"message": "Record deleted successfully."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Failed to delete record: {str(e)}"}
        )


@app.post("/run-pipeline")
async def run_pipeline_endpoint():
    global pipeline_running
    pipeline_running = True

    thread = threading.Thread(target=pipeline)
    thread.start()

    thread.join()

    pipeline_running = False
    return JSONResponse(status_code=200, content={"message": "Pipeline is running."})


# Endpoint to check pipeline status
@app.get("/pipeline-status")
async def get_pipeline_status():
    return {"running": pipeline_running}
