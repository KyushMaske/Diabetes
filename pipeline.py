import os
import shutil
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn import svm
from joblib import dump
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from dotenv import load_dotenv
import pickle


load_dotenv()

BACKUP_DIR = "model_backups"


os.makedirs(BACKUP_DIR, exist_ok=True)


DATABASE_URL = os.getenv("DATABASE_URL")

# Path to save the updated model and scaler
MODEL_PATH = "diabetes_model.pkl"
SCALER_PATH = "scaler.pkl"
CSV_PATH = "diabetes.csv"
RESULTS_FILE = "model_results.txt"


def save_results(train_accuracy, test_accuracy, class_report):
    """Save the model results to a text file, appending if it already exists, along with a timestamp."""
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Format: YYYY-MM-DD HH:MM:SS
    with open(RESULTS_FILE, "a") as f:  
        f.write(f"Timestamp: {timestamp}\n")  
        f.write(
            f"Training accuracy: {train_accuracy:.4f}\n"
        )  # Format to 4 decimal places
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")  
        f.write("\nClassification Report:\n")
        f.write(class_report)
        f.write("\n" + "-" * 40 + "\n")  
    print(f"Results appended to {RESULTS_FILE}")


def clear_backup_directory():
    """Delete all files in the backup directory."""
    for filename in os.listdir(BACKUP_DIR):
        file_path = os.path.join(BACKUP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted old backup file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def fetch_data_from_db(CSV_PATH):
    """Fetch data from PostgreSQL and save it to a CSV file."""
    conn = None  
    try:
        
        conn = psycopg2.connect(DATABASE_URL)

        # Define the SQL query to fetch data
        query = """
        SELECT pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, 
               diabetes_pedigree_function, age, prediction 
        FROM diabetes_predictions;
        """


        df = pd.read_sql_query(query, conn)
        print("Data fetched from database:")
        print(df)  

        df["prediction"] = df["prediction"].apply(lambda x: 1 if x == "Diabetic" else 0)

        df.rename(
            columns={
                "pregnancies": "Pregnancies",
                "glucose": "Glucose",
                "blood_pressure": "BloodPressure",
                "skin_thickness": "SkinThickness",
                "insulin": "Insulin",
                "bmi": "BMI",
                "diabetes_pedigree_function": "DiabetesPedigreeFunction",
                "age": "Age",
                "prediction": "Outcome",
            },
            inplace=True,
        )

        print(df)

       
        if not os.path.isfile(CSV_PATH):
            df.to_csv(
                CSV_PATH, mode="w", index=False, header=True
            ) 
        else:
            df.to_csv(
                CSV_PATH, mode="a", index=False, header=False
            )  

        print(
            f"{len(df)} records fetched from the database and appended to {CSV_PATH}."
        )
        return df  

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  

    finally:
        # Close the database connection
        if conn:
            conn.close()


def train_new_model(csv_path):
    """Train a new model using the data fetched from the database."""
    # Splitting data into features (X) and labels (y)
    df = pd.read_csv(csv_path)

    length = len(df)
    print(f"The number of rows in the DataFrame is: {length}")


    X = df.drop(columns="Outcome")  # Drop 'Outcome' to create features
    y = df["Outcome"]  # Label column

    # Data Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE to handle class imbalance in the training set
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    clear_backup_directory()

    # Backup the old model if it exists
    if os.path.exists(MODEL_PATH):
        # Generate a new filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"diabetes_model_{timestamp}.pkl"
        shutil.move(MODEL_PATH, os.path.join(BACKUP_DIR, backup_filename))
        print(f"Old model backed up as {backup_filename}")


    model = svm.SVC(kernel="linear", class_weight="balanced")
    model.fit(X_train, y_train)


    X_train_prediction = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, X_train_prediction)
    print(f"Training accuracy: {train_accuracy}")


    X_test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, X_test_prediction)
    print(f"Test accuracy: {test_accuracy}")


    class_report = classification_report(y_test, X_test_prediction)
    print(class_report)


    save_results(train_accuracy, test_accuracy, class_report)


    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("New model trained and saved.")



def flush_database():
    """Delete all records from the predictions table."""
    conn = None  # Initialize conn to None for safe closure
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        delete_query = (
            "DELETE FROM diabetes_predictions;"  
        )
        cursor.execute(delete_query)
        conn.commit()
        cursor.close()
        print("Database flushed (all records deleted).")

    except Exception as e:
        print(f"An error occurred while flushing the database: {e}")

    finally:
        if conn:
            conn.close()


def pipeline():
    print(f"Pipeline started at {datetime.now()}")


    fetch_data_from_db(CSV_PATH)


    train_new_model(CSV_PATH)


    flush_database()


# # Scheduler to run the pipeline every hour
# scheduler = BlockingScheduler()
# scheduler.add_job(pipeline, 'interval', minutes=1)  # Set to run every minute for testing

# if __name__ == "__main__":
#     # Start the scheduler
#     print("Starting the scheduled pipeline.")
#     scheduler.start()
