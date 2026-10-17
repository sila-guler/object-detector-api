# Object Detector API

Google Colab'da eğitilen ResNet50 tabanlı nesne tespit modelini (Faces / Motorbikes / airplanes) FastAPI ile Docker konteyneri üzerinde sunar.

## Çalıştırma

```bash
git clone <repo-adresi>
cd <repo-klasörü>
docker-compose up
```

Servis `http://localhost:7001` adresinde çalışır. Etkileşimli dokümantasyon: `http://localhost:7001/docs`

## Kullanım

```bash
curl -X POST -F "file=@ornek.jpg" http://localhost:7001/predict
```

Örnek yanıt:

```json
{"label": "airplanes", "probability": 0.9999, "bbox": [43, 27, 356, 134]}
```

`bbox`, tahmin edilen kutunun orijinal görüntüdeki piksel koordinatlarıdır (x1, y1, x2, y2).
