import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import insightface
from insightface.app import FaceAnalysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

MOCK_DATABASE = [
    {
        "username": "@ahmet_yilmaz",
        "platform": "instagram",
        "profile_url": "https://instagram.com",
        "embedding": np.random.randn(512)
    },
    {
        "username": "@ece_tekin",
        "platform": "tiktok",
        "profile_url": "https://tiktok.com",
        "embedding": np.random.randn(512)
    }
]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profil Finder AI</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #fff; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        .card { background: #1e293b; border-radius: 16px; padding: 24px; width: 100%; max-width: 440px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h2 { margin-top: 0; color: #38bdf8; font-size: 26px; }
        .upload-area { border: 2px dashed #38bdf8; padding: 25px 15px; border-radius: 12px; cursor: pointer; margin: 20px 0; background: rgba(56, 189, 248, 0.02); }
        input[type="file"] { display: none; }
        button { background: #38bdf8; color: #0f172a; border: none; padding: 14px; font-weight: bold; border-radius: 8px; cursor: pointer; width: 100%; font-size: 16px; }
        .result-item { background: #334155; margin-top: 12px; padding: 14px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; }
        .badge { background: #22c55e; color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: bold; }
    </style>
</head>
<body>

<div class="card">
    <h2>Profil Finder AI</h2>
    <p style="color: #94a3b8; font-size: 14px;">Fotoğraf yükleyin, eşleşen sosyal medya profillerini tarayalım.</p>
    
    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <span id="fileName">📁 Fotoğraf Seçmek İçin Dokunun</span>
        <input type="file" id="fileInput" accept="image/*" onchange="showFileName()">
    </div>

    <button onclick="uploadImage()">Aramayı Başlat</button>

    <div id="results" style="margin-top: 20px;"></div>
</div>

<script>
    function showFileName() {
        const input = document.getElementById('fileInput');
        if (input.files.length > 0) {
            document.getElementById('fileName').innerText = "📄 " + input.files[0].name;
        }
    }

    async function uploadImage() {
        const input = document.getElementById('fileInput');
        const resultsDiv = document.getElementById('results');

        if (input.files.length === 0) {
            alert("Lütfen bir fotoğraf seçin!");
            return;
        }

        resultsDiv.innerHTML = "<p style='color: #38bdf8;'>Yüz taranıyor ve analiz ediliyor...</p>";

        const formData = new FormData();
        formData.append("file", input.files[0]);

        try {
            const response = await fetch("/search-face", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.status === "error") {
                resultsDiv.innerHTML = `<p style="color: #ef4444;">${data.message}</p>`;
                return;
            }

            resultsDiv.innerHTML = "";
            data.results.forEach(res => {
                resultsDiv.innerHTML += `
                    <div class="result-item">
                        <div style="text-align: left;">
                            <strong>${res.username}</strong>
                            <div style="font-size: 12px; color: #94a3b8;">${res.platform.toUpperCase()}</div>
                        </div>
                        <span class="badge">%${res.match_score}</span>
                    </div>
                `;
            });

        } catch (err) {
            resultsDiv.innerHTML = "<p style='color: #ef4444;'>Sunucuyla bağlantı kurulamadı.</p>";
        }
    }
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTML_CONTENT

@app.post("/search-face")
async def search_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if len(faces) == 0:
        return {"status": "error", "message": "Görselde yüz tespit edilemedi."}

    target_embedding = faces[0].embedding

    results = []
    for item in MOCK_DATABASE:
        score = cosine_similarity(target_embedding, item["embedding"])
        match_percentage = round(float((score + 1) / 2) * 100, 2)
        
        results.append({
            "username": item["username"],
            "platform": item["platform"],
            "profile_url": item["profile_url"],
            "match_score": match_percentage
        })

    results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    return {"status": "success", "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
