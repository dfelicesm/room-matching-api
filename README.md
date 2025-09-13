# Room Matching API

This project shows how to match hotel room names between an internal catalog and supplier data.

The user should provide a hotel code (an `lp_id`) and the the name of the room to be matched. Then, the API will return all the potential matches that are included in the internal catalog scored for similarity (0-100).

## Project structure

- `notebooks/`: contains Jupyter notebooks that explain the thought process and experiments behind feature extraction and preprocessing used for room matching.
- `src/`: contains the source code for the API and supporting modules.
- `data/`: contains example input data and the provided internal catalog (`referance_rooms-1737378184366`).

## Running the API

### Locally
#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Start the API
```bash
uvicorn src.api:app --reload
```

#### 3. Test the API
Open your browser at `http://127.0.0.1:8000/docs` to test the `/match`endpoint using Swagger. There are some examples of rooms and hotels in the data folder.

### Using Docker
#### 1. Build the image
```bash
docker build -t room-matching-api .
```

#### 2. Run the container
```bash
docker run -p 8000:8000 room-matching-api
```
The API will be again available at `http://127.0.0.1:8000/docs`.
