# Домашнее задание по курсу "ML в продакшене"
## Использование
### Обучение модели
```
python3 ml_project/train.py                — по умолчанию обучается логистическая регрессия            
python3 ml_project/train.py model=busting  — обучить градиентный бустинг
python3 ml_project/train.py model=logreg   — обучить логистическую регрессию
```
### Получение предсказаний
```
python3 ml_project/predict.py artifacts/model.pkl <features path> <prediction path>
```
## Online inference
### Локальная сборка и запуск docker-образа
```
docker build -t hw2:v1 .
docker run -p 8000:8000 hw2:v1
```
### Импорт и запуск docker-образа из dockerhub
```
docker pull ruslan16/ml_in_prod_2022:latest
docker run -p 8000:8000 ruslan16/ml_in_prod_2022:latest
```
### Запуск скрипта для работы с сервисом
```
python online_inference/client.py
```
## Структура проекта
```
├── artifacts           <- Место сохранения обученной модели 
├── configs             <- Конфигурационные файлы
├── ml_project          
│   ├── data            <- Загрузка и чтение данных
│   ├── features        <- Предобработка данных
│   ├── models          <- Интструменты работы с моделью
│   ├── raw             <- Место сохранения датасета для обучения
│   ├── predict.py      <- Получение предсказаний на основе обученной модели
│   └── train.py        <- Обучение модели
├── notebooks           <- Разведочный анализ данных
├── outputs             <- Предсказанные метки
├── reports             <- Метрики
├── requirements.txt    <- Зависимости
└── tests               <- Тесты
```
## Описание проекта
В проекте реализована модель обнаружения заболевания сердца. Модель обучена на датасете [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci). Доступно две конфигурации для обучения: логистическая регрессия и градиентный бустинг. Для классификации, оценки результатов, разделения и трансформации данных использовалась боблиотека [sklearn](https://scikit-learn.org/stable/index.html).
