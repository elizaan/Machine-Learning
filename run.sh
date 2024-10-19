cd "Decision Tree"
python src/main.py --data car --depth 6 --et 
python src/main.py --data car --depth 6 --gi
python src/main.py --data car --depth 6 --me
python src/main.py --data data --depth 16 --et
python src/main.py --data data --depth 16 --gi
python src/main.py --data data --depth 16 --me

cd "Ensemble Learning"
python src/main.py --data data --adaboost
python src/main.py --data data --bagging
python src/main.py --data data --rf --subset_size 2
python src/main.py --data data --rf --subset_size 4
python src/main.py --data data --rf --subset_size 6
python src/main.py --data data --bv --bagging
python src/main.py --data data --bv --rf --subset_size 2
python src/main.py --data data --bv --rf --subset_size 4
python src/main.py --data data --bv --rf --subset_size 6

cd "Linear Regression"
python src/main.py
