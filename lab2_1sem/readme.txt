Файлы *.py распределены по номерам пунктов и подпунктов,
в соответствии с подзадачами данной лабораторки (вроде однозначное соответствие)
Запуск: python3 ...

Программы *.py работают, если в одной папке с ними лежат скачанные файлы данных,
это сделано специально, чтобы не зависеть от доступности / недоступности урлов

Также прикрепляю файл lab02_trees.ipynb (в котором тот же код, что и в файлах *.py)
Здесь все ссылки на файлы данных даны в виде урлов, но почему - то students.csv по
урлу не парсится и выдаёт ошибку, поэтому его скачал в папку с lab02_trees.ipynb
и считываю локально, соответственно, для корректного запуска, его также нужно скачать.

Кроме того, при запуске lab02_trees.ipynb не всегда читались библиотеки,
которые были загружены в предыдущих пунктах, и пришлось каждый пункт перепроходить
с помощью кнопки run (|>|). Не знаю это просто был какой - то сбой,
или библиотеки надо всегда дублировать.

В остальном вроде все работает.

python3 lab2_1_1-3.py ../../machinelearning_data/lab2_1sem/students.csv
python3 lab2_1_4-5.py ../../machinelearning_data/lab2_1sem/agaricus-lepiota.data
python3 lab2_1_6-7.py ../../machinelearning_data/lab2_1sem/agaricus-lepiota.data ../../machinelearning_data/lab2_1sem/tic-tac-toe.data ../../machinelearning_data/lab2_1sem/car.data ../../machinelearning_data/lab2_1sem/nursery.data
python3 lab2_2.py ../../machinelearning_data/lab2_1sem/winequality-red.csv
python3 lab2_3.py ../../machinelearning_data/lab2_1sem/winequality-red.csv
