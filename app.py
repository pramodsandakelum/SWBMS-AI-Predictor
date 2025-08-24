from fastapi import FastAPI, Response
import pandas as pd
import psycopg2
import joblib
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database environment variables
DB_PARAMS = {
    "dbname": os.environ.get("POSTGRES_DB", "postgres"),
    "user": os.environ.get("POSTGRES_USER", "postgres.haecgyxeccjjwsyomvuy"),
    "password": os.environ.get("POSTGRES_PASSWORD", "Postgre1234"),
    "host": os.environ.get("POSTGRES_HOST", "aws-0-ap-southeast-1.pooler.supabase.com"),
    "port": os.environ.get("POSTGRES_PORT", "6543")
}

# ----------------- Core Training + Prediction -----------------
def train_and_predict():
    """Train per-bin models and save predictions for next weekend"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Fetch all readings
        df = pd.read_sql("SELECT * FROM readings", conn)
        if df.empty:
            print("No readings found. Skipping prediction.")
            cur.close()
            conn.close()
            return "No readings found"

        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
        df = df.sort_values(['bin_id','recorded_at'])

        # Create lag features
        for lag in [1,2,3]:
            df[f'weight_t-{lag}'] = df.groupby('bin_id')['weight_kg'].shift(lag)
            df[f'fullness_t-{lag}'] = df.groupby('bin_id')['fullness_percent'].shift(lag)

        df['hour'] = df['recorded_at'].dt.hour
        df['dayofweek'] = df['recorded_at'].dt.dayofweek
        df = df.dropna()

        features = [f'weight_t-{lag}' for lag in [1,2,3]] + \
                   [f'fullness_t-{lag}' for lag in [1,2,3]] + ['hour','dayofweek']

        # Next weekend dates
        today = datetime.utcnow()
        next_saturday = today + timedelta((5-today.weekday()) % 7)
        next_sunday = today + timedelta((6-today.weekday()) % 7)

        predictions_to_save = []

        for bin_id in df['bin_id'].unique():
            bin_data = df[df['bin_id']==bin_id]
            X = bin_data[features]
            y = bin_data['weight_kg']

            # Train LightGBM
            model = LGBMRegressor()
            model.fit(X, y)
            joblib.dump(model, f"lgbm_model_bin_{bin_id}.pkl")

            # Future dataframe
            future_df = pd.DataFrame({
                'hour': [0,6,12,18]*2,
                'dayofweek': [5]*4 + [6]*4
            })

            last_row = bin_data.iloc[-1]
            for lag in [1,2,3]:
                future_df[f'weight_t-{lag}'] = last_row[f'weight_t-{lag}']
                future_df[f'fullness_t-{lag}'] = last_row[f'fullness_t-{lag}']

            pred_weight = model.predict(future_df[features])

            timestamps = [next_saturday + timedelta(hours=h) for h in range(0,24,6)] + \
                         [next_sunday + timedelta(hours=h) for h in range(0,24,6)]

            for t, w in zip(timestamps, pred_weight):
                predictions_to_save.append((
                    bin_id,
                    float(w),
                    t.date(),
                    datetime.utcnow()
                ))

        # Save predictions
        insert_query = """
        INSERT INTO predictions (bin_id, predicted_weight_kg, prediction_month, created_at)
        VALUES (%s, %s, %s, %s)
        """
        cur.executemany(insert_query, predictions_to_save)
        conn.commit()
        cur.close()
        conn.close()
        print(f"[{datetime.utcnow()}] Predictions saved successfully.")
        return "Predictions saved"

    except Exception as e:
        print(f"Error in train_and_predict: {e}")
        return str(e)

# ----------------- Scheduler -----------------
scheduler = BackgroundScheduler()
scheduler.add_job(train_and_predict, 'cron', day_of_week='sun', hour=0, minute=0)
scheduler.start()

# ----------------- API Endpoints -----------------
@app.get("/latest_predictions")
def latest_predictions():
    """Return latest prediction per bin"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("""
            SELECT p.bin_id, p.predicted_weight_kg, p.prediction_month, p.created_at
            FROM predictions p
            INNER JOIN (
                SELECT bin_id, MAX(created_at) AS max_created
                FROM predictions
                GROUP BY bin_id
            ) sub ON p.bin_id = sub.bin_id AND p.created_at = sub.max_created
            ORDER BY p.bin_id;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        predictions = [
            {
                "device_id": str(r[0]),
                "predicted_weight": r[1],
                "predicted_for": r[2].isoformat(),
                "prediction_timestamp": r[3].isoformat()
            }
            for r in rows
        ]
        return {"status": "success", "predictions": predictions}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/prediction_history")
def prediction_history():
    """Return all historical predictions"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("""
            SELECT bin_id, predicted_weight_kg, prediction_month, created_at
            FROM predictions
            ORDER BY created_at DESC;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        history = [
            {
                "device_id": str(r[0]),
                "predicted_weight": r[1],
                "predicted_for": r[2].isoformat(),
                "prediction_timestamp": r[3].isoformat()
            }
            for r in rows
        ]
        return {"status": "success", "predictions": history}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------- NEW Endpoints -----------------
@app.post("/train")
def manual_train():
    """Manually trigger model training and prediction"""
    result = train_and_predict()
    return {"status": "success", "message": result}

@app.get("/ping")
@app.head("/ping")
def ping(response: Response):
    """Keep-alive endpoint for Render uptime monitor"""
    return {"alive": True, "time": datetime.utcnow().isoformat()}
