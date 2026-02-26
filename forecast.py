import pandas as pd
from prophet import Prophet

def generate_forecast(df):

    monthly = df.groupby(["Year", "Month"]).size().reset_index(name="count")

    monthly["ds"] = pd.to_datetime(
        monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
    )

    monthly = monthly.rename(columns={"count": "y"})
    monthly = monthly[["ds", "y"]]

    model = Prophet()
    model.fit(monthly)

    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    return forecast