import pickle
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker


class DiabetesModel:
    def __init__(self, model_path, scaler_path, db_url):
        # Load the model and scaler
        with open(model_path, "rb") as model_file:
            self.model = pickle.load(model_file)

        with open(scaler_path, "rb") as scaler_file:
            self.scaler = pickle.load(scaler_file)

        # Set up the database connection
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        # Define the table structure for storing predictions
        self.metadata = MetaData()
        self.diabetes_table = Table(
            "diabetes_predictions",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("pregnancies", Integer),
            Column("glucose", Float),
            Column("blood_pressure", Float),
            Column("skin_thickness", Float),
            Column("insulin", Float),
            Column("bmi", Float),
            Column("diabetes_pedigree_function", Float),
            Column("age", Integer),
            Column("prediction", String),
        )

        # Create the table if it doesn't exist
        self.metadata.create_all(self.engine)

    def predict(self, input_data):

        scaled_data = self.scaler.transform(input_data)

        prediction = self.model.predict(scaled_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return result

    def save_prediction(self, data, result):

        try:
            insert_statement = self.diabetes_table.insert().values(
                pregnancies=data["pregnancies"],
                glucose=data["glucose"],
                blood_pressure=data["blood_pressure"],
                skin_thickness=data["skin_thickness"],
                insulin=data["insulin"],
                bmi=data["bmi"],
                diabetes_pedigree_function=data["diabetes_pedigree_function"],
                age=data["age"],
                prediction=result,
            )
            self.session.execute(insert_statement)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
