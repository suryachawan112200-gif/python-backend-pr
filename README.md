# Python Backend Project

## Overview
This project is a FastAPI backend application that serves as a foundation for building a trading application. It includes essential dependencies and a simple endpoint to get started.

## Project Structure
```
python-backend-project
├── src
│   └── main.py
├── requirements.txt
├── .venv
└── README.md
```

## Setup Instructions

1. **Clone the repository** (if applicable):
   ```
   git clone <repository-url>
   cd python-backend-project
   ```

2. **Create a virtual environment**:
   ```
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the FastAPI application, execute the following command:
```
uvicorn src.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` in your browser to access the interactive API documentation.

## Dependencies
The project requires the following Python packages:
- fastapi
- uvicorn
- pandas
- numpy
- requests
- yfinance
- python-binance
- alpha-vantage
- redis
- pydantic
- easyocr

## License
This project is licensed under the MIT License.