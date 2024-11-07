cd "Decision Tree"
python src/main.py --data car --depth 6 --et 
python src/main.py --data car --depth 6 --gi
python src/main.py --data car --depth 6 --me
python src/main.py --data bank --depth 16 --et
python src/main.py --data bank --depth 16 --gi
python src/main.py --data bank --depth 16 --me

cd "Ensemble Learning"
python src/main.py --data bank --adaboost
python src/main.py --data bank --bagging
python src/main.py --data bank --rf --subset_size 2
python src/main.py --data bank --rf --subset_size 4
python src/main.py --data bank --rf --subset_size 6
python src/main.py --data bank --bv --bagging
python src/main.py --data bank --bv --rf --subset_size 2
python src/main.py --data bank --bv --rf --subset_size 4
python src/main.py --data bank --bv --rf --subset_size 6

cd "Linear Regression"
python src/main.py
