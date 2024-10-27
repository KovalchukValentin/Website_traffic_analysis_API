from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Збереження даних у пам'яті
traffic_data_storage = []

# Модель для вхідних даних трафіку
class TrafficDataInput(BaseModel):
    timestamp: str
    visitors: int
    page_views: int
    bounce_rate: float
    avg_time_on_page: float


# Маршрут для додавання даних трафіку
@app.post("/traffic-data/")
async def add_traffic_data(data: List[TrafficDataInput]):
    global traffic_data_storage
    data_dicts = [item.dict() for item in data]
    traffic_data_storage.extend(data_dicts)
    return {"message": "Traffic data added successfully", "data_count": len(traffic_data_storage)}


# Маршрут для отримання всіх даних
@app.get("/traffic-data/")
async def get_traffic_data():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data found")
    return {"traffic_data": traffic_data_storage}


# Маршрут для отримання статистики трафіку
@app.get("/traffic-statistics/")
async def get_traffic_statistics():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    # Перетворення на DataFrame
    df = pd.DataFrame(traffic_data_storage)

    # Основні метрики для трафіку
    stats = {
        "total_visitors": float(df["visitors"].sum()),
        "total_page_views": float(df["page_views"].sum()),
        "avg_bounce_rate": float(df["bounce_rate"].mean()),
        "avg_time_on_page": float(df["avg_time_on_page"].mean())
    }

    return {"statistics": stats}


# Кореляція між показниками
@app.get("/traffic-correlation/")
async def get_traffic_correlation():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)

    # Обчислення кореляції між різними показниками
    correlation_matrix = df[["visitors", "page_views", "bounce_rate", "avg_time_on_page"]].corr()

    return {"correlation_matrix": correlation_matrix.to_dict()}


# Тренд аналіз
@app.get("/traffic-trends/")
async def get_traffic_trends():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Аналіз тренду для відвідувачів і переглядів сторінок
    visitors_trend = np.polyfit(df.index, df["visitors"], 1)[0]
    page_views_trend = np.polyfit(df.index, df["page_views"], 1)[0]

    trend_analysis = {
        "visitors_trend": "increasing" if visitors_trend > 0 else "decreasing",
        "page_views_trend": "increasing" if page_views_trend > 0 else "decreasing"
    }

    return {"trend_analysis": trend_analysis}


# Зведена статистика (мінімум, максимум, медіана)
@app.get("/traffic-summary-statistics/")
async def get_summary_statistics():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)

    summary_stats = {
        "min_visitors": float(df["visitors"].min()),
        "max_visitors": float(df["visitors"].max()),
        "median_visitors": float(df["visitors"].median()),
        "min_page_views": float(df["page_views"].min()),
        "max_page_views": float(df["page_views"].max()),
        "median_page_views": float(df["page_views"].median()),
        "min_bounce_rate": float(df["bounce_rate"].min()),
        "max_bounce_rate": float(df["bounce_rate"].max()),
        "median_bounce_rate": float(df["bounce_rate"].median()),
        "min_avg_time_on_page": float(df["avg_time_on_page"].min()),
        "max_avg_time_on_page": float(df["avg_time_on_page"].max()),
        "median_avg_time_on_page": float(df["avg_time_on_page"].median())
    }

    return {"summary_statistics": summary_stats}


# Аналіз сезонності (групування за днями тижня)
@app.get("/traffic-seasonality/")
async def get_traffic_seasonality():
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Групування за днями тижня
    df["day_of_week"] = df["timestamp"].dt.day_name()
    seasonality = df.groupby("day_of_week")[["visitors", "page_views"]].mean().to_dict()

    return {"seasonality_analysis": seasonality}


# Маршрут для прогнозування кількості відвідувачів на основі історичних даних
@app.get("/predict-visitors/")
async def predict_visitors(days: int = 7):
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Перетворення дат у числа для регресії
    df['day_num'] = (df["timestamp"] - df["timestamp"].min()).dt.days
    X = np.array(df["day_num"]).reshape(-1, 1)
    y = np.array(df["visitors"])

    # Модель лінійної регресії
    model = LinearRegression()
    model.fit(X, y)

    # Прогноз на наступні кілька днів
    future_days = np.array([df["day_num"].max() + i for i in range(1, days + 1)]).reshape(-1, 1)
    predictions = model.predict(future_days)

    future_dates = pd.date_range(df["timestamp"].max(), periods=days + 1, freq='D')[1:]

    prediction_results = [{"date": str(date.date()), "predicted_visitors": pred} for date, pred in zip(future_dates, predictions)]

    return {"predictions": prediction_results}


# Маршрут для ковзного середнього по кількості відвідувачів
@app.get("/moving-average-visitors/")
async def moving_average_visitors(window: int = 3):
    if not traffic_data_storage:
        raise HTTPException(status_code=404, detail="No traffic data available")

    df = pd.DataFrame(traffic_data_storage)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")

    # Обчислення ковзного середнього для відвідувачів
    df["moving_average_visitors"] = df["visitors"].rolling(window=window).mean()

    return {"moving_average": df[["timestamp", "moving_average_visitors"]].dropna().to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
