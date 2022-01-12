@REM learning rate variation
@REM python .\hw1-q4.py mlp -learning_rate 0.001 -hidden_sizes 100 -dropout 0.3 -activation relu -optimizer sgd -layers 2
@REM python .\hw1-q4.py mlp -learning_rate 0.001 -hidden_sizes 100 -dropout 0.3 -activation relu -optimizer sgd -layers 3
@REM python .\hw1-q4.py mlp -learning_rate 0.01 -hidden_sizes 100 -dropout 0.3 -activation relu -optimizer sgd
@REM python .\hw1-q4.py mlp -learning_rate 0.1 -hidden_sizes 100 -dropout 0.3 -activation relu -optimizer sgd
@REM @REM learning hidden_sizes
@REM python .\hw1-q4.py mlp -learning_rate 0.01 -hidden_sizes 200 -dropout 0.3 -activation relu -optimizer sgd
@REM @REM learning droput
@REM python .\hw1-q4.py mlp -learning_rate 0.01 -hidden_sizes 100 -dropout 0.5 -activation relu -optimizer sgd
@REM @REM learning activation
@REM python .\hw1-q4.py mlp -learning_rate 0.01 -hidden_sizes 100 -dropout 0.3 -activation tanh -optimizer sgd
@REM @REM learning optimizer
@REM python .\hw1-q4.py mlp -learning_rate 0.01 -hidden_sizes 100 -dropout 0.3 -activation relu -optimizer adam

python .\hw1-q4.py logistic_regression -learning_rate 0.1 
python .\hw1-q4.py logistic_regression -learning_rate 0.01
python .\hw1-q4.py logistic_regression -learning_rate 0.001