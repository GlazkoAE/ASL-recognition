# ASL-recognition
[American Sing Language](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) recognition project for MLOps ITMO task

## Что сделано
1. Валидация данных в исходном датасете (ASL_raw)

2. Подготовка данных и их загрузка на clearml в датасет ASL. Датасет разделен на ASL_raw и ASL в целях имитации feature store, где разделены данные на предобработанные и сырые размеченные. В датасете ASL_raw изображения не разбросаны на train/val/test и могут быть любого размера.

3. Обучение модели. Написано на PyTorch с логгированием через tensorboard. После обучения выгружаются .onnx и .txt с классами как артефакты, на агенте они при этом удаляются.

4. Написан инференс на triton, причем pbtxt конфиги автоматически переписываются под новую модель. Её данные берутся из задачи обучения. Вслучае изменения модели, количества классов или размера картинок, все подгрузится автоматически в конфиги.

5. При открытии pull request код проверяется через flake8.

## Что не сделано, но будет
1. Сервинг модели. На момент завершения курса не успел написать парсер для prometheus. При сервинге модели планируется раз в n минут мониторить состояние модели и оповещать разработчика (telegram / slack / почта) в случае преодоления заданных порогов характеристик модели на сервере. Также планируется организовать доступ к веб странице с удобно представленными данными по модели.
2. Демонстрация удаленного доступа к серверу.
3. Загрузка матрицы ошибок после обучения модели.
4. Выбор лучшей модели после каждого нового обучения из данных clearml.
5. Запуск пайплайна обучения при слиянии в веток (к примеру, лияние в test / develop).

## Запуск сервера и клиента
1. Сервер
```
docker build --tag triton-server <repo_root>/triton/server/
docker run --name server --rm -p8000:8000 -p8001:8001 -p8002:8002 triton-server tritonserver --model-repository=/models
```

2. Клиент
```
docker build --tag triton-client <repo_root>/triton/client/
docker run --name client --net=host --rm -it -v /path/to/images:/test_images triton-client /bin/sh
python3 client.py --image /test_images/<image_file>
```
