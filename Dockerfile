# 1️⃣ Utiliser l'image Python comme base
FROM python:3.12

# 2️⃣ Définir le répertoire de travail
WORKDIR /app

# 3️⃣ Copier les fichiers dans le conteneur
COPY . /app

# 4️⃣ Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# 5️⃣ Exposer le port utilisé par FastAPI
EXPOSE 8000

# 6️⃣ Démarrer l'API avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
