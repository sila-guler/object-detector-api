from fastapi import FastAPI, File, HTTPException, UploadFile

from app.predictor import predict

app = FastAPI(title="Object Detector API")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        return predict(await file.read())
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçerli bir görüntü dosyası yükleyin")
