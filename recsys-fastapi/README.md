# FastAPI User Action Logging Project

This project is a FastAPI application designed to log user actions to a PostgreSQL database and retrieve the 10 most recent logs for user behavior analysis. The application is structured to facilitate easy development and maintenance.

## Project Structure

```
recsys-fastapi
├── app
│   ├── main.py                # Entry point of the FastAPI application
│   ├── api
│   │   ├── __init__.py        # Marks the api directory as a package
│   │   └── logs.py            # API endpoints for logging user actions
│   ├── core
│   │   ├── __init__.py        # Marks the core directory as a package
│   │   └── config.py          # Configuration settings for the application
│   ├── db
│   │   ├── __init__.py        # Marks the db directory as a package
│   │   ├── session.py         # Manages database session and connection
│   │   ├── models.py          # Defines database models for logging
│   │   └── crud.py            # Functions for creating and retrieving logs
│   ├── schemas
│   │   └── logs.py            # Pydantic schemas for validating log data
│   ├── services
│   │   └── logger.py          # Logging service for user actions
│   └── utils
│       └── scheduler.py       # Scheduler for fetching recent logs
├── tests
│   └── test_logs.py           # Unit tests for logging functionality
├── alembic
│   ├── env.py                 # Database migrations setup
│   └── versions               # Migration scripts directory
├── .env                       # Environment variables for the application
├── Dockerfile                 # Instructions to build a Docker image
├── requirements.txt           # Project dependencies
├── pyproject.toml            # Project configuration
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd recsys-fastapi
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the PostgreSQL database and update the `.env` file with your database credentials.

## Usage

1. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

## Features


## Contributing
 
- FE calling guide: [`docs/FE_API_GUIDE.md`](docs/FE_API_GUIDE.md)

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.