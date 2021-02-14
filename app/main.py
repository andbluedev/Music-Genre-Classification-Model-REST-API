from fastapi import FastAPI


app = FastAPI(
    title="Model API for Music Genre Classification",
    version="1.0"
)


@app.get('/live')
async def live():
    return {"status": "OK"}
