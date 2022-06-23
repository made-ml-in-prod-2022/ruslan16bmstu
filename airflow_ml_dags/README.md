## Использование
```
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export MODEL_PATH="/data/models/selected/model.pkl"
docker compose up --build
```