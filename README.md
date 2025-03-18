# House Price Prediction API

This project provides a machine learning model to predict house prices based on input features.

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `python app.py`

## Usage
Send a POST request to `/predict` with JSON input:
```
{
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "bedrooms": 3,
  "bathrooms": 2,
  "zipcode": "98052"
}
```
Response:
```
{
  "predicted_price": 450000
}
```

## Docker
To run in Docker:
```
docker build -t house-price-api .
docker run -p 5000:5000 house-price-api
